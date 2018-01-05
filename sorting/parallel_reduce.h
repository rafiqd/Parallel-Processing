// Penn Parallel Primitives Library
// Author: Prof. Milo Martin
// University of Pennsylvania
// Spring 2010

#ifndef PPP_REDUCE_H
#define PPP_REDUCE_H

#include "ppp.h"
#include "Task.h"
#include "TaskGroup.h"

namespace ppp {
  
  
  namespace internal {

  
  template <typename T>  
  class ReduceTask: public ppp::Task {
  
  public:
		ReduceTask( T* array, int64_t left, int64_t right, int64_t grainsize, atomic<T> *sum ) { 
			m_left = left;
			m_right = right;
			m_grainsize = grainsize;
			m_array = array;
			m_sum = sum;
			}
		
		void execute(){
			assert(m_left < m_right);

			if (m_right-m_left <= 1) {
				return;
			}
		
			if (m_right-m_left < m_grainsize) {
				T tempsum;
				tempsum = T(0);
				int64_t i;
				for( i = m_left; i < m_right; ++i){
					tempsum = tempsum + m_array[i];
				}
				m_sum->fetch_and_add( tempsum );
				return;
			}

			ppp::TaskGroup tg;
			int64_t pivot = (m_right - m_left)/2 + m_left;
			
			
			ReduceTask t1( m_array, m_left, pivot , m_grainsize, m_sum);
			ReduceTask t2( m_array, pivot, m_right, m_grainsize, m_sum);
			tg.spawn(t1);
			tg.spawn(t2);
			tg.wait();
		}

		private:
			int64_t m_left;
			int64_t m_right;
			int64_t m_grainsize;
			T* m_array;
			atomic<T> *m_sum;
	};
  }
  
  template <typename T>
  extern inline
  T parallel_reduce(T* array, int64_t start, int64_t end, int64_t grainsize=0)
  {
	
	if (grainsize == 0) {
      grainsize = (start-end+1) / (get_thread_count()*4);
    }
    PPP_DEBUG_MSG("parallel_reduce grainsize: " + to_string(grainsize));
    atomic<T> asum;
	asum.set(0);
    internal::ReduceTask<T> t(array, start, end, grainsize, &asum);
    t.execute();
	
	return asum.get();
    T sum;
    sum = T(0);
    return sum;
	
  }
}

#endif
