#pragma once

#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <vector>
#include <queue>
#include <thread>


class ThreadPool
{
private:
	struct data
	{
		std::mutex mtx_;
		std::condition_variable cond_;
		bool is_shutdown_ = false;
		std::queue<std::function<void()>> tasks_;
	};
	std::shared_ptr<data> data_;

public:

	inline ThreadPool(unsigned short thread_count) : data_(std::make_shared<data>())
	{
		for (int i = 0; i < thread_count; i++)
		{
			std::thread([data = data_]
				{
					std::unique_lock<std::mutex> lk(data->mtx_);
					for (;;)
					{
						if (!data->tasks_.empty())
						{
							auto current = std::move(data->tasks_.front());
							data->tasks_.pop();
							current();
							lk.lock();
						}
						else if (data->is_shutdown_)
						{
							break;
						}
						else
						{
							data->cond_.wait(lk);
						}
					}
				}).detach();
		}
	}

	ThreadPool() = default;
	ThreadPool(ThreadPool&&) = default;

	inline ~ThreadPool()
	{
		if ((bool)data_)
		{
			{
				std::lock_guard<std::mutex> lk(data_->mtx_);
				data_->is_shutdown_ = true;
			}
			data_->cond_.notify_all();
		}
	}

	template<class F, class... Args>
	auto commit(F&& f, Args&&... args) ->std::future<decltype(f(args...))>
	{
		using RetType = decltype(f(args...)); // typename std::result_of<F(Args...)>::type, 函数 f 的返回值类型
		auto task = std::make_shared<std::packaged_task<RetType()> >(
			std::bind(std::forward<F>(f), std::forward<Args>(args)...)
			);
		std::future<RetType> future = task->get_future();
		{
			std::lock_guard<std::mutex> lk(data_->mtx_);
			data_->tasks_.emplace([task] {(*task)(); });
		}
		data_->cond_.notify_one();

		return future;
	}
};
