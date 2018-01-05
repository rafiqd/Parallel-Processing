// Penn Parallel Primitives Library
// Author: Prof. Milo Martin
// University of Pennsylvania
// Spring 2010

#include "ppp.h"
#include "TaskGroup.h"

namespace ppp {

  namespace internal {
    // Task-based scheduling variables
    TaskQueue* g_queues_ptr = NULL;
    atomic<int> g_stop_counter;  
  }

  using namespace internal;

  void TaskGroup::spawn(Task& t) {
    assert(g_queues_ptr != NULL);
	TaskQueue& queue = g_queues_ptr[ get_thread_id() ];  // ASSIGNMENT: use per-thread task queue with "get_thread_id()"
    m_wait_counter.fetch_and_inc();
    t.setCounter(&m_wait_counter);
    queue.enqueue(&t);
  }
  
  void process_tasks(const atomic<int>* counter)
  {
	int num;
    TaskQueue& queue = g_queues_ptr[ get_thread_id() ];  // ASSIGNMENT: use per-thread task queue with "get_thread_id()"

	//printf("queue size: %d\n", 
    while ( (num = counter->get()) != 0) {
      PPP_DEBUG_EXPR(queue.size());
       
      // Dequeue from local queue
      Task* task = queue.dequeue();

/*********************************************/
    // ASSIGNMENT: add task stealing
	int max = 0;
	int size;
	int index = -1;
	if(queue.size() == 0){
		// get index of which queue to steal from -- largest queue
		for(int i = 0; i < internal::s_thread_count; ++i){
			size = g_queues_ptr[ i ].size();
			if(  size > max){
				index = i;
				max = size;
			}
		}
		// have index to steal from now
		if( index != -1){
			//printf("I am thread %d stealing from thread %d\n", get_thread_id(), index);
			queue.enqueue( g_queues_ptr[ index ].steal() );
		}  
	}
/***********************************************/

      if (task != NULL) {
        task->execute(); // overloaded method
        task->post_execute(); // cleanup, method of base class
      }
    }
  }
}

