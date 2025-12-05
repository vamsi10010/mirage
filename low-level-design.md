# Low-Level Design: Priority-Based Scheduling for Mirage LLM Serving

## 1. Problem Statement / Scenario

### Multi-User Serving Environment
In a real-world Large Language Model (LLM) serving environment, the system receives inference requests from multiple users simultaneously. These requests are heterogeneous in nature:
*   **Different Importance**: A system prompt or a premium user's request should take precedence over a free-tier user's request.
*   **Different Urgency**: Some tasks (e.g., real-time chat) have strict latency Service Level Objectives (SLOs), while others (e.g., offline batch summarization) are delay-tolerant.

### The Challenge
The current Mirage scheduling implementation treats all requests as equal and processes them in a strict First-In-First-Out (FIFO) order based on their initialization index. In a constrained resource environment (limited GPU memory and compute), this leads to:
1.  **Head-of-Line Blocking**: Low-priority long-running requests can block high-priority urgent requests.
2.  **SLO Violations**: Urgent requests may miss their deadlines because the scheduler is unaware of time constraints.
3.  **Inefficient Resource Allocation**: The system cannot make trade-offs between throughput and latency based on business value.

## 2. Existing Approach

### Current Scheduling Algorithm
The current scheduling logic resides in `include/mirage/persistent_kernel/persistent_kernel.cuh`, specifically within the `prepare_next_batch` device function.

**Mechanism:**
1.  **Capacity Model**: The system pre-allocates memory for a fixed capacity of requests (`total_num_requests`), indexed from `0` to `N-1`.
2.  **State Tracking**: It uses a simple counter `*config.next_request_id` to track the next request to process.
3.  **Batch Filling**:
    *   The kernel iterates through the active batch slots (`MPK_MAX_NUM_BATCHED_REQUESTS`).
    *   If a slot is free, it assigns the request ID pointed to by `next_request_id`.
    *   It increments `next_request_id`.

**Code Reference (`persistent_kernel.cuh`):**
```cpp
// Current FIFO Logic
while (num_reqs < MPK_MAX_NUM_BATCHED_REQUESTS && ...) {
    int next_request_id = *config.next_request_id; // <--- Strictly sequential
    if (next_request_id >= config.total_num_requests) {
        break;
    }
    config.request_ids[num_reqs] = next_request_id;
    // ... setup request ...
    *config.next_request_id = next_request_id + 1; // <--- Increment
}
```

**Limitations in Multi-User Context:**
If User A submits a low-priority batch job (Requests 0-99) and User B submits a high-priority chat message (Request 100), User B must wait for all 100 previous requests to be scheduled. The kernel has no mechanism to "skip" to Request 100.

## 3. Project Requirements

1.  **Priority Support**: The system must support at least 3 distinct priority levels (e.g., High, Medium, Low).
2.  **Deadline Awareness**: The system must track an absolute deadline for each request derived from its SLO.
3.  **Preemptive-like Scheduling**: The scheduler must be able to select a high-priority request from the waiting pool even if it arrived later (higher ID) than low-priority requests.
4.  **GPU-Resident Logic**: The scheduling decision must happen entirely on the GPU to maintain the performance benefits of the megakernel architecture.
5.  **Demonstration**: A simulation script must prove that high-priority requests bypass low-priority ones.

## 4. Design Details

### 4.1. Data Structure Updates
We need to introduce new metadata tensors to track the state and properties of each request slot.

**File**: `include/mirage/persistent_kernel/runtime_header.h`

```cpp
struct RuntimeConfig {
    // ... existing fields ...
    
    // NEW: Scheduling Metadata
    // 0: FREE, 1: WAITING (Ready to run), 2: RUNNING, 3: FINISHED
    int *request_status; 
    
    // 0: Low, 1: Medium, 2: High
    int *priorities;
    
    // Absolute timestamp (e.g., arrival_time + SLO_duration)
    long long *deadlines; 
};
```

### 4.2. Integration with Mirage (Python & C++)
We must pipe these new tensors from the PyTorch host environment to the CUDA kernel.

**File**: `python/mirage/persistent_kernel.py`
*   Update `__init__` to accept `priorities`, `deadlines`, and `request_status` tensors.
*   Update `compile` to append these tensor pointers to the `meta_tensors` list passed to the C++ initialization function.

**File**: `include/mirage/persistent_kernel/persistent_kernel.cuh`
*   Update `init_persistent_kernel` to cast the `void*` arguments from Python back to specific pointers in `RuntimeConfig`.

### 4.3. Priority and Deadline Based Scheduling Logic
We will replace the `next_request_id` counter with a **Scan-and-Select** algorithm inside `prepare_next_batch`.

**Algorithm: Priority-Based Earliest Deadline First (P-EDF)**
1.  Iterate through all request slots (`0` to `total_num_requests`).
2.  Filter for requests where `request_status == WAITING`.
3.  Select the request with the **Highest Priority**.
4.  Tie-breaker: Select the request with the **Earliest Deadline** (Smallest value).

**Code Snippet (`persistent_kernel.cuh`):**

```cpp
__device__ __forceinline__ bool prepare_next_batch(RuntimeConfig const &config) {
    // ... (Existing cleanup logic) ...

    // Fill empty batch slots
    while (num_reqs < MPK_MAX_NUM_BATCHED_REQUESTS && 
           num_tokens < MPK_MAX_NUM_BATCHED_TOKENS) {
        
        int best_candidate = -1;
        int max_priority = -1;
        long long min_deadline = 9223372036854775807LL; // Max Long Long

        // SCAN PHASE: Find the best WAITING request
        // Optimization: For large N, we can use a bitmask or hierarchical bitmap 
        // to skip non-waiting requests, but linear scan is acceptable for N < 1000.
        for (int r = 0; r < config.total_num_requests; r++) {
            if (config.request_status[r] == 1) { // WAITING
                int p = config.priorities[r];
                long long d = config.deadlines[r];

                bool is_better = false;
                if (p > max_priority) {
                    is_better = true;
                } else if (p == max_priority) {
                    if (d < min_deadline) {
                        is_better = true;
                    }
                }

                if (is_better) {
                    max_priority = p;
                    min_deadline = d;
                    best_candidate = r;
                }
            }
        }

        // ASSIGN PHASE
        if (best_candidate != -1) {
            config.request_ids[num_reqs] = best_candidate;
            config.request_status[best_candidate] = 2; // Mark as RUNNING
            
            // ... (Initialize tokens for best_candidate) ...
            
            num_reqs++;
        } else {
            // No more waiting requests
            break;
        }
    }
    
    // ... (Update pointers) ...
}
```

### 4.4. Custom Heuristic
The design allows for pluggable heuristics. The P-EDF logic above can be modified to include "Aging" to prevent starvation.

**Aging Heuristic:**
```cpp
long long current_time = clock64(); // GPU clock
// Effective Priority = Base Priority + (Wait Time / Aging Factor)
int effective_priority = config.priorities[r] + 
    (current_time - config.arrival_times[r]) / AGING_THRESHOLD;
```
*Note: For the initial implementation, we will stick to strict P-EDF.*

### 4.5. Multi-User Environment Demo
We will create `demo/qwen3/demo_priority.py`. Unlike `demo_chat.py` which is interactive, this will be a simulation.

**Simulation Logic:**
1.  **Setup**: Initialize kernel with capacity for 100 requests.
2.  **Scenario**:
    *   Requests 0-4: Low Priority (Batch job), Status=WAITING.
    *   Request 5: High Priority (VIP User), Status=WAITING.
    *   Request 6: Medium Priority, Status=WAITING.
3.  **Execution**: Run the kernel for one step.
4.  **Verification**: Check `step_tensor`. Request 5 should have advanced (step > 0), while Requests 0-4 might still be at step 0 (if batch size < 6).

## 5. Evaluation

We will evaluate the implementation using the `demo_priority.py` script.

**Metrics:**
1.  **Correctness**: Does the High Priority request *always* get scheduled before Low Priority requests when resources are contended?
2.  **Latency**: Measure the "Time to First Token" (TTFT) for High vs. Low priority requests in a saturated system.
3.  **Overhead**: Measure the execution time of `prepare_next_batch` with the new scanning logic vs. the old O(1) logic. (Expected to be negligible for N < 1000).

## 6. Timeline (5 Days)

| Day | Task | Details |
| :--- | :--- | :--- |
| **Day 1** | **Core Data Structures** | 1. Modify `runtime_header.h` to add `priorities`, `deadlines`, `request_status`.<br>2. Update `persistent_kernel.cuh` initialization logic to reset these arrays. |
| **Day 2** | **Python Integration** | 1. Update `PersistentKernel` class in `persistent_kernel.py` to accept new tensors.<br>2. Update C++ `init_persistent_kernel` to map Python pointers to C++ config.<br>3. Verify compilation passes. |
| **Day 3** | **Scheduler Logic** | 1. Implement the Scan-and-Select P-EDF logic in `prepare_next_batch`.<br>2. Replace the `next_request_id` dependency.<br>3. Handle state transitions (WAITING -> RUNNING). |
| **Day 4** | **Simulation Demo** | 1. Create `demo_priority.py`.<br>2. Implement the simulation scenario (High priority arriving after Low priority).<br>3. Add verification logic to check execution order. |
| **Day 5** | **Testing & Tuning** | 1. Run the simulation.<br>2. Debug any race conditions or logic errors.<br>3. Measure overhead.<br>4. Finalize documentation. |
