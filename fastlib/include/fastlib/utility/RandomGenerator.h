#pragma once

#include <fastlib/FastLibDef.h>

#include <utility>
#include <random>
#include <functional>
#include <mutex>
#include <chrono>
#include <numeric>


namespace fast {

	template <typename E, typename D, class ...TArgs >
	struct RandomGenerator {
		
		/*
			Forwards distribution parameters
		*/
		RandomGenerator(TArgs && ... args)
			: _distribtion(std::forward<TArgs>(args)...)
		{}

		/*
			Copy assignment sans mutex
		*/
		RandomGenerator& operator = (const RandomGenerator &g) {
			_engine = g._engine;
			_distribtion = g._distribtion;
			return *this;
		}

		/*
			Copy constructor sans mutex
		*/
		RandomGenerator(const RandomGenerator &g) {
			_engine = g._engine;
			_distribtion = g._distribtion;
		}		


		void seedByTime() {
			auto timept = std::chrono::high_resolution_clock::now();
			_engine.seed(timept.time_since_epoch().count());
		}

		/*
			Returns next value from the random generator
		*/
		typedef typename D::result_type result_type;
		result_type next() {
			std::lock_guard<std::mutex> lock(_mutex);
			return _distribtion(_engine);
		}

		D & getDistribution() {
			return _distribtion;
		}

	private:
		std::mutex _mutex;
		E _engine;
		D _distribtion;
	};

	using RNGNormal = RandomGenerator<
		std::default_random_engine,
		std::normal_distribution<float>,
		float, float>;
	using RNGUniformInt = RandomGenerator<
		std::default_random_engine,
		std::uniform_int_distribution<int>,
		int, int>;
	using RNGUniformFloat = RandomGenerator<
		std::default_random_engine,
		std::uniform_real_distribution<float>,
		float, float>;	
	using RNGLogNormal = RandomGenerator<
		std::default_random_engine,
		std::lognormal_distribution<float>,
		float, float>;



	FAST_EXPORT void exec(int index, const std::function<void(void)> & f);

	template<typename ... Fargs>
	void exec(int index, const std::function<void(void)> & f, Fargs ... args) {
		if (index == 0)
			f();
		else
			exec(index - 1, args ...);
	}

	/*
		Randomly chooses between passed functions and executes one
	*/
	template<typename ... Largs>
	void choose(RNGUniformInt & rnd, Largs ... args) {
		int index = rnd.next() % (sizeof...(args));
		exec(index, args ...);
	}



	/*
		Returns -1 or 1, 50% chance
	*/
	FAST_EXPORT int randomBi(RNGUniformInt & rnd);

	template <class T>
	int chooseIndexFromVector(
		const std::vector<T> & v,
		float prob
	) {
		T sum = std::accumulate(v.begin(), v.end(), T(0));
		struct Pair {
			float val;
			int index;
		};
		
		std::vector<Pair> cdf;
		for (int i = 0; i < v.size(); i++) {
			//float prev = (i == 0) ? 0.0f : v[i - 1];
			cdf.push_back({ float(v[i]) / sum, i });
		}

		const auto cmp = [](const Pair & a, const Pair & b) { return a.val < b.val; };
		const auto cmpF = [](const Pair & a, const float & b) { return a.val < b; };
		std::sort(cdf.begin(), cdf.end(), cmp);

		for (int i = 1; i < v.size(); i++) {
			cdf[i].val += cdf[i - 1].val;
		}

		auto it = std::lower_bound(cdf.begin(), cdf.end(), prob, cmpF);

		return it->index;
	}

	

}