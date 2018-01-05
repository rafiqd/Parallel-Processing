// Penn Parallel Primitives Library
// Author: Prof. Milo Martin
// University of Pennsylvania
// Spring 2010

#ifndef PPP_FOR_H
#define PPP_FOR_H

#include "ppp.h"
#include "Task.h"
#include "TaskGroup.h"

namespace ppp {

  namespace internal {
  template <typename T>  
  class ComputeTask: public ppp::Task {
  public:
		ComputeTask( int64_t left, int64_t right, int64_t grainsize, T* functor) { 
			m_left = left;
			m_right = right;
			m_grainsize = grainsize;
			m_functor = functor;
			}
			void execute(){
			assert(m_left < m_right);

			if (m_right-m_left <= 1) {
				return;
			}
		
			if (m_right-m_left < m_grainsize) {
				m_functor->calculate(m_left, m_right); 
				return;
			}
		
			ppp::TaskGroup tg;
			int64_t pivot = (m_right - m_left)/2 + m_left;
			
			
			ComputeTask t1( m_left, pivot , m_grainsize, m_functor);
			ComputeTask t2( pivot, m_right, m_grainsize, m_functor);
			tg.spawn(t1);
			tg.spawn(t2);
			tg.wait();
		}
	  
		private:
			int64_t m_left;
			int64_t m_right;
			int64_t m_grainsize;
			T* m_functor;
	};
  }
  template <typename T>
  extern inline
  void parallel_for(int64_t start, int64_t end, T* functor, int64_t grainsize=0)
  {
    // ASSIGNMENT: make this parallel via recursive divide and conquer
	
    if (grainsize == 0) {
      grainsize = (start-end+1) / (get_thread_count()*4);
    }
	
	internal::ComputeTask<T> t(start, end, grainsize, functor);
    t.execute();
	
    PPP_DEBUG_MSG("parallel_sort done");
  }
}

#endif
