#include "runner.hpp"
#include <fstream>
#include <queue>
#include <deque>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <sstream>

#define LEADER_PERCENTAGE 0.05
#define COMPLETED_TASK_ID_MPI_TAG 100
#define SENT_BACK_TASKS_MPI_TAG 101
#define FOLLOWER_IDLE_STATUS_MPI_TAG 102
#define ASSIGNING_TASKS_MPI_TAG 200
#define COMPLETION_SIGNAL_TAG 201


// Modified from main.cpp line 24 to get MPI struct implementation
MPI_Datatype create_MPI_TASK_T() {
    MPI_Datatype MPI_TASK_T;
    constexpr int task_t_num_blocks = 8;
    constexpr int task_t_lengths[task_t_num_blocks] = {1, 1, 1, 1, 1, 1, 4, 4};
    constexpr MPI_Aint task_t_displs[task_t_num_blocks] = {
        offsetof(struct task_t, id),
        offsetof(struct task_t, gen),
        offsetof(struct task_t, type),
        offsetof(struct task_t, arg_seed),
        offsetof(struct task_t, output),
        offsetof(struct task_t, num_dependencies),
        offsetof(struct task_t, dependencies),
        offsetof(struct task_t, masks)};
    MPI_Datatype task_t_types[task_t_num_blocks] = {MPI_UINT32_T, MPI_INT, MPI_INT, MPI_UINT32_T, MPI_UINT32_T, MPI_INT, MPI_UINT32_T, MPI_UINT32_T};

    MPI_Type_create_struct(task_t_num_blocks, task_t_lengths, task_t_displs, task_t_types,
                           &MPI_TASK_T);

    MPI_Type_commit(&MPI_TASK_T);

    return MPI_TASK_T;
}

/*
 * ==========================================
 * | [START] OK TO MODIFY THIS FILE [START] |
 * ==========================================
 */
bool is_request_complete(MPI_Request& request) {
  int request_complete;
  MPI_Test(&request, &request_complete, MPI_STATUS_IGNORE);
  return request_complete != 0;
}

/**
 * Change this to get back a completed task
*/
void check_for_completed_task_notifications(
  MPI_Datatype MPI_TASK_T,
  const std::vector<int>& followers,
  std::unordered_map<uint32_t, std::pair<bool, task_t>>& completed_tasks
) {
  for (int follower: followers) {
    MPI_Status status;
    int flag;

    MPI_Iprobe(follower, COMPLETED_TASK_ID_MPI_TAG, MPI_COMM_WORLD, &flag, &status);
    if (flag) {
      task_t received_task;
      MPI_Recv(&received_task, 1, MPI_TASK_T, follower, COMPLETED_TASK_ID_MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (completed_tasks.find(received_task.id) != completed_tasks.end()) {
        completed_tasks[received_task.id].first = true;
        completed_tasks[received_task.id].second = received_task;
      }
    }
  }
}

/**
 * Check for sent back tasks from followers
*/
void check_for_sent_back_tasks(
  MPI_Datatype MPI_TASK_T,
  const std::vector<int>& followers,
  std::vector<task_t>& task_receive_buffer
) {
  for (int follower: followers) {
    MPI_Status status;
    int flag;

    MPI_Iprobe(follower, SENT_BACK_TASKS_MPI_TAG, MPI_COMM_WORLD, &flag, &status);
    if (flag) {
      int count;
      MPI_Get_count(&status, MPI_TASK_T, &count);
      std::vector<task_t> received_tasks(count);
      MPI_Recv(received_tasks.data(), count, MPI_TASK_T, follower, SENT_BACK_TASKS_MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      for (const auto& task : received_tasks) {
        if (task.id) {
          task_receive_buffer.push_back(task);
        }
      }
    }
  }
}

/**
 * Check for status updates from followers
*/
void check_follower_status(
  const std::vector<int>& followers,
  std::deque<int>& idle_followers
) {
  for (int follower: followers) {
    MPI_Status status;
    int flag;

    MPI_Iprobe(follower, FOLLOWER_IDLE_STATUS_MPI_TAG, MPI_COMM_WORLD, &flag, &status);
    if (!flag) {
      continue;
    }
    int idle_status;
    MPI_Recv(&idle_status, 1, MPI_INT, follower, FOLLOWER_IDLE_STATUS_MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Check if the follower is already in the idle queue
    bool is_already_idle = std::find(idle_followers.begin(), idle_followers.end(), follower) != idle_followers.end();

    if (idle_status && !is_already_idle) {
      idle_followers.push_back(follower);
    }
  }
}

/**
 * Select tasks to distribute from queued tasks according to dependencies
 * Change this to check dependencies based on arg_seed. For now just do bit | according to output of task
 * At the end, check if resultant arg seed is positive (to clarify this)
*/
void select_tasks_to_distribute(
  std::list<task_t>& queued_tasks,
  std::queue<task_t>& tasks_to_send,
  std::unordered_map<uint32_t, std::pair<bool, task_t>>& completed_tasks,
  bool& just_received_initial_tasks
) {
  for (auto it = queued_tasks.begin(); it != queued_tasks.end();) {
    task_t& task = *it;

    bool dependencies_met = true;

    uint32_t arg_seed = task.arg_seed;

    if (!just_received_initial_tasks) {
      for (int i = 0; i < task.num_dependencies; ++i) {
        if (completed_tasks.find(task.dependencies[i]) == completed_tasks.end()) {
          dependencies_met = false;
          break;
        }

        if (!completed_tasks[task.dependencies[i]].first) {
          dependencies_met = false;
          break;
        }

        uint32_t output = completed_tasks[task.dependencies[i]].second.output;
        for (auto& mask: completed_tasks[task.dependencies[i]].second.masks) {
          output &= mask;
        }
        arg_seed |= output;
      }
    }

    if (dependencies_met) {
      task.arg_seed = arg_seed;
      tasks_to_send.push(task);
      it = queued_tasks.erase(it);
    } else {
      ++it;
    }
  }

  just_received_initial_tasks = false;
}

/**
 * Only send to idle followers. Returns vector of vector of tasks to distribute.
 * Idea: if a follower is in idle_followers, either they sent idle notification
 * or no tasks were sent out yet
*/
void prepare_task_distribution(
  std::vector<std::vector<task_t>>& task_distribution,
  std::queue<task_t>& tasks_to_send,
  std::deque<int>& idle_followers,
  const std::vector<int>& followers,
  const int num_followers
) {
  if (idle_followers.empty()) {
    return;
  }

  // Determine the number of tasks per follower to balance the load
  int total_tasks = tasks_to_send.size();
  int num_idle_followers = idle_followers.size();
  int min_tasks_per_follower = total_tasks / num_idle_followers;
  int extra_tasks = total_tasks % num_idle_followers;

  while (!tasks_to_send.empty()) {
    int follower_rank = idle_followers.front();
    idle_followers.pop_front();

    // Find the index of the follower in the followers vector
    auto it = std::find(followers.begin(), followers.end(), follower_rank);
    if (it == followers.end()) {
      // Follower rank not found in the followers vector
      continue;
    }
    int follower_idx = std::distance(followers.begin(), it);

    // Calculate the number of tasks to assign to this follower
    int tasks_for_this_follower = min_tasks_per_follower;
    if (extra_tasks > 0) {
      tasks_for_this_follower++;
      extra_tasks--;
    }

    // Assign the calculated number of tasks
    for (int i = 0; i < tasks_for_this_follower && !tasks_to_send.empty(); i++) {
      task_distribution[follower_idx].push_back(tasks_to_send.front());
      tasks_to_send.pop();
    }
  }
}

/**
 * Sends tasks and awaits for tasks to be sent. We send batches of tasks
*/
void send_tasks_in_task_distribution(
  MPI_Datatype MPI_TASK_T,
  std::vector<std::vector<task_t>>& task_distribution,
  std::vector<MPI_Request>& send_requests,
  std::unordered_map<uint32_t, std::pair<bool, task_t>>& task_completion_map,
  const std::vector<int>& followers,
  int num_followers
) {
  for (int i = 0; i < num_followers; ++i) {
    if (!task_distribution[i].empty()) {
      MPI_Isend(task_distribution[i].data(), task_distribution[i].size(), MPI_TASK_T, followers[i], ASSIGNING_TASKS_MPI_TAG, MPI_COMM_WORLD, &send_requests[i]);

      for (const auto& task : task_distribution[i]) {
        task_completion_map[task.id] = std::make_pair(false, task);
      }
    }
  }

  // Add wait here on to ensure no undefined behavior.
  // From tests, this was not too necessary but currently this only works
  // because of interconnectivity between nodes on cluster. To be correct in any
  // environment, ensure that we wait for all tasks sent back before removing tasks
  for (int i = 0; i < num_followers; ++i) {
    if (!task_distribution[i].empty()) {
      MPI_Wait(&send_requests[i], MPI_STATUSES_IGNORE);
      task_distribution[i].clear();
    }
  }
}

void run_leader_process(int rank, const std::vector<task_t>& send_list, MPI_Datatype MPI_TASK_T, const std::vector<int>& followers) {
  int num_followers = followers.size();
  std::vector<task_t> task_receive_buffer;
  std::list<task_t> queued_tasks;
  std::queue<task_t> tasks_to_send;
  std::vector<std::vector<task_t>> task_distribution(num_followers);
  // Make this a deque.
  std::deque<int> idle_followers;
  std::unordered_set<uint32_t> completed_tasks;
  // Changed to have value of pair(bool, task_t). first means whether task was completed and
  // task is valid to check against. Second is the task itself.
  std::unordered_map<uint32_t, std::pair<bool, task_t>> task_completion_map;
  std::vector<MPI_Request> send_requests(followers.size(), MPI_REQUEST_NULL); // To track requests

  for (int follower : followers) {
    idle_followers.push_back(follower);
  }

  for (auto& task: send_list) {
    queued_tasks.push_back(task);
  }

  bool just_received_initial_tasks = true;

  while (true) {
    select_tasks_to_distribute(queued_tasks, tasks_to_send, task_completion_map, just_received_initial_tasks);
    prepare_task_distribution(task_distribution, tasks_to_send, idle_followers, followers, num_followers);
    send_tasks_in_task_distribution(MPI_TASK_T, task_distribution, send_requests, task_completion_map, followers, num_followers);

    check_follower_status(followers, idle_followers);
    check_for_completed_task_notifications(MPI_TASK_T, followers, task_completion_map);
    check_for_sent_back_tasks(MPI_TASK_T, followers, task_receive_buffer);

    for (const auto& task : task_receive_buffer) {
      queued_tasks.push_back(task);
    }
    task_receive_buffer.clear();

    bool can_terminate = queued_tasks.empty() &&
      idle_followers.size() == num_followers &&
      !task_completion_map.empty() && std::all_of(
        task_completion_map.begin(),
        task_completion_map.end(),
        [](const auto& entry) { return entry.second.first; }
      );

    if (can_terminate) {
      int completion_signal = 1;
      for (int follower: followers) {
        MPI_Send(&completion_signal, 1, MPI_INT, follower, COMPLETION_SIGNAL_TAG, MPI_COMM_WORLD);
      }
      break;
    }
  }
}

bool check_for_termination_signal(int leader_rank) {
  int flag;
  MPI_Status status;
  int termination_signal;

  MPI_Iprobe(leader_rank, COMPLETION_SIGNAL_TAG, MPI_COMM_WORLD, &flag, &status);
  
  if (flag) {
    MPI_Recv(&termination_signal, 1, MPI_INT, leader_rank, COMPLETION_SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return true;
  }
  
  return false;
}

void receive_tasks_from_leader(
  MPI_Datatype MPI_TASK_T,
  std::vector<task_t>& task_receive_buffer,
  int leader_rank
) {
  MPI_Status status;
  int flag;

  MPI_Iprobe(leader_rank, ASSIGNING_TASKS_MPI_TAG, MPI_COMM_WORLD, &flag, &status);

  if (flag) {
    int count;
    MPI_Get_count(&status, MPI_TASK_T, &count);
    std::vector<task_t> received_tasks(count);
    MPI_Recv(received_tasks.data(), count, MPI_TASK_T, leader_rank, ASSIGNING_TASKS_MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    for (const auto& task : received_tasks) {
        task_receive_buffer.push_back(task);
    }
  }
}

void send_completion_to_leader(
  int rank,
  int leader_rank,
  task_t& task,
  MPI_Request& send_request,
  MPI_Datatype MPI_TASK_T
) {
  if (!is_request_complete(send_request)) {
    return;
  }

  MPI_Isend(&task, 1, MPI_TASK_T, leader_rank, COMPLETED_TASK_ID_MPI_TAG, MPI_COMM_WORLD, &send_request);
}

void send_generated_tasks_to_leader(
  int rank,
  int leader_rank,
  std::vector<task_t>& generated_tasks,
  MPI_Datatype MPI_TASK_T,
  MPI_Request& send_request
) {
  if (!is_request_complete(send_request)) {
    return;
  }

  if (!generated_tasks.empty()) {    
    MPI_Isend(generated_tasks.data(), generated_tasks.size(), MPI_TASK_T, leader_rank, SENT_BACK_TASKS_MPI_TAG, MPI_COMM_WORLD, &send_request);
  }
}

void notify_status_to_leader(
  int rank,
  int leader_rank,
  bool is_idle,
  MPI_Request& send_request
) {
  if (!is_request_complete(send_request)) {
    return;
  }

  int idle_status = is_idle ? 1 : 0;
  MPI_Isend(&idle_status, 1, MPI_INT, leader_rank, FOLLOWER_IDLE_STATUS_MPI_TAG, MPI_COMM_WORLD, &send_request);
}

void run_follower_process(
  int rank,
  metric_t &stats,
  MPI_Datatype MPI_TASK_T,
  int leader_rank
) {
  std::vector<task_t> task_receive_buffer;
  std::queue<task_t> queued_tasks;
  std::vector<std::vector<task_t>> generated_tasks_array;

  std::vector<MPI_Request> completion_requests;
  std::vector<MPI_Request> generated_tasks_array_requests;
  MPI_Request status_request = MPI_REQUEST_NULL;

  while (true) {
    if (check_for_termination_signal(leader_rank)) {
      break;
    }

    receive_tasks_from_leader(MPI_TASK_T, task_receive_buffer, leader_rank);

    for (const auto& task : task_receive_buffer) {
      queued_tasks.push(task);
      std::vector<task_t> generated_tasks(Nmax);
      generated_tasks_array.push_back(generated_tasks);
      completion_requests.push_back(MPI_REQUEST_NULL);
      generated_tasks_array_requests.push_back(MPI_REQUEST_NULL);
    }
    task_receive_buffer.clear();

    bool task_queue_was_empty = queued_tasks.empty();

    int i = 0;
    while (!queued_tasks.empty()) {
      task_t task = queued_tasks.front();
      queued_tasks.pop();
      int num_new_tasks = 0;
      execute_task(stats, task, num_new_tasks, generated_tasks_array[i]);
      send_completion_to_leader(rank, leader_rank, task, completion_requests[i], MPI_TASK_T);
      send_generated_tasks_to_leader(rank, leader_rank, generated_tasks_array[i], MPI_TASK_T, generated_tasks_array_requests[i]);
      ++i;
    }

    if (!task_queue_was_empty) {
      notify_status_to_leader(rank, leader_rank, true, status_request);
    }

    // Add wait all here on to ensure no undefined behavior.
    // From tests, this was not too necessary but currently this only works
    // because of interconnectivity between nodes on cluster. To be correct in any
    // environment, ensure that we wait for all task requests and completion requests to be complete
    // and then clear requests and tasks
    MPI_Waitall(completion_requests.size(), completion_requests.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(generated_tasks_array_requests.size(), generated_tasks_array_requests.data(), MPI_STATUSES_IGNORE);
    MPI_Wait(&status_request, MPI_STATUSES_IGNORE);

    generated_tasks_array.clear();
    completion_requests.clear();
    generated_tasks_array_requests.clear();

  }
}

/**
 * Changed to make 0 a leader as well.
*/
void set_up_node_graph(std::vector<int>& index, std::vector<int>& edges, int num_procs, int num_leaders) {
  std::vector<std::vector<int>> graph_edges(num_procs);

  // now only have leaders and followers
  for (int i = num_leaders; i < num_procs; ++i) {
    int leader = i % num_leaders;
    graph_edges[leader].push_back(i);
    graph_edges[i].push_back(leader);
  }

  // Convert to index and edges
  for (int i = 0; i < num_procs; ++i) {
    index[i] = edges.size();
    edges.insert(edges.end(), graph_edges[i].begin(), graph_edges[i].end());
  }
}

void run_all_tasks(int rank, int num_procs, metric_t &stats, params_t &params) {
  std::vector<task_t> send_list;
  std::ifstream istrm(params.input_path, std::ios::binary);
  MPI_Datatype MPI_TASK_T = create_MPI_TASK_T();
  int count;
  istrm >> count;

  if (rank == 0) {
    for (int i = 0; i < count; i++) {
      task_t task;
      int type;
      istrm >> type >> task.arg_seed;
      task.type = static_cast<TaskType>(type);
      task.id = task.arg_seed;
      task.gen = 0;
      send_list.push_back(task);
    }
  }

  MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);
  int num_leaders = std::max(1, static_cast<int>(std::ceil(LEADER_PERCENTAGE * (num_procs))));
  num_leaders = std::min(num_leaders, count);
  std::vector<int> index(num_procs), edges;
  set_up_node_graph(index, edges, num_procs, num_leaders);
  std::vector<int> neighbors;
  int start_idx = index[rank];
  int end_idx = (rank == num_procs - 1) ? edges.size() : index[rank + 1];

  for (int i = start_idx; i < end_idx; ++i) {
    neighbors.push_back(edges[i]);
  }

  std::vector<task_t> to_send;

  if (rank == 0) {
    if (num_leaders == 1) {
      to_send = send_list;
    } else {
      std::vector<MPI_Request> send_requests(num_leaders - 1);
      std::vector<std::vector<task_t>> tasks_by_leader(num_leaders);
      int i = 0;
      for (const auto& task : send_list) {
        tasks_by_leader[i].push_back(task);
        i = (i + 1) % num_leaders;
      }
      for (int j = 0; j < num_leaders; ++j) {
        if (j == 0) {
          for (auto& task: tasks_by_leader[j]) {
            to_send.push_back(task);
          }
        } else {
          MPI_Isend(tasks_by_leader[j].data(), tasks_by_leader[j].size(), MPI_TASK_T, j, 1, MPI_COMM_WORLD, &send_requests[j-1]);
        }
      }
      // Add back wait all here on to ensure no undefined behavior.
      MPI_Waitall(num_leaders - 1, send_requests.data(), MPI_STATUSES_IGNORE);
    }
  } else if (rank < num_leaders) {
    while (true) {
      MPI_Status status;
      int flag;
      MPI_Iprobe(0, 1, MPI_COMM_WORLD, &flag, &status);
      if (flag) {
        int count;
        MPI_Get_count(&status, MPI_TASK_T, &count);
        std::vector<task_t> received_tasks(count);
        MPI_Recv(received_tasks.data(), count, MPI_TASK_T, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (auto& task : received_tasks) {
          to_send.push_back(task);
        }
        break;
      }
    }
  }

  if (rank < num_leaders) {
    run_leader_process(rank, to_send, MPI_TASK_T, neighbors);
  } else {
    int leader_rank = neighbors.front();
    run_follower_process(rank, stats,MPI_TASK_T,leader_rank);
  }
}
