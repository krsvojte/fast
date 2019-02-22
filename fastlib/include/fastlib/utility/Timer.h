#pragma once

#include <chrono>
#include <map>
#include <sstream>
#include <algorithm>

namespace fast {

	class Timer {
		public:

			Timer(bool autoStart = false) {				
				if(autoStart)
					start();
			}

			void start() {
				_running = true;
				_start = std::chrono::system_clock::now();
			}

			void stop() {
				_end = std::chrono::system_clock::now();
				_running = false;
			}

			float timeMs() {				
				return time() * 1000.0f;
			}

			//Returns time in seconds
			float time() {
				if (_running) stop();
				std::chrono::duration<float> duration = _end - _start;
				return duration.count();
			}

		private: 
			std::chrono::time_point<std::chrono::system_clock> _start;
			std::chrono::time_point<std::chrono::system_clock> _end;
			bool _running;

	};


	class Profiler {
		struct Event {
			float totalTime = 0.0f;
			int n = 0;

			void add(float t) {
				totalTime += t;
				n++;
			}

			float avg() const {
				return totalTime / n;
			}
		};
	
	
	public:
		Profiler(){
			_timerTotal.start();
		}

		void stop() {
			_timerTotal.stop();
		}

		
		void add(const std::string & name, float t) {
			_events[name].add(t);
		}


		std::map<std::string, Event> & events() {
			return _events;
		}


		std::string summary() {
			std::stringstream ss;

			using pair_t = std::pair<std::string, Event>;
			std::vector<pair_t> sortedEvents;
			for (auto & it : _events) {
				sortedEvents.push_back({ it.first, it.second });
			}

			std::sort(sortedEvents.begin(), sortedEvents.end(), [](const pair_t & a, const pair_t & b) { return a.second.avg() > b.second.avg(); });
			
			
			for (auto & it : sortedEvents) {
				auto & ev = it.second;
				ss << it.first << ": \n";
				ss << "\tN:\t\t" << ev.n << "\n";
				ss << "\tTotal:\t\t" << ev.totalTime << "s\n";
				ss << "\tAvg:\t\t" << ev.avg() << "s\n";
				ss << "\tPercent:\t\t" << (ev.totalTime / _timerTotal.time()) * 100.0f << "%\n";
			}

			ss << "PROFILER TOTAL: " << _timerTotal.time() << "s " << "\n";

			return ss.str();
		}


	private:
		std::map<std::string, Event> _events;
		
		

		Timer _timerTotal;

		

	};




	
	


}