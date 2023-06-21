#ifndef __CONCURRENCEQUEUE_H__
#define __CONCURRENCEQUEUE_H__
#include <mutex>
#include <condition_variable>
#include <deque>
#include <queue>
#include <memory>

template<typename DATATYPE, typename SEQUENCE = std::deque<DATATYPE>>
class ConcurrenceQueue {
public:
    ConcurrenceQueue() = default;
    ~ConcurrenceQueue() = default;
    ConcurrenceQueue & operator= (const ConcurrenceQueue &) = delete;
    /*
    ConcurrenceQueue(const ConcurrenceQueue & other) {
        std::lock_guard<std::mutex> lg(other.m_mutex);
        m_data = other.m_data;
    }
    ConcurrenceQueue(ConcurrenceQueue &&) = delete;
    */
    bool empty() const {
        std::lock_guard<std::mutex> lg(m_mutex);
        return m_data.empty();
    }
    int size(){
        std::lock_guard<std::mutex>lg(m_mutex);
        return m_data.size();
    }
    int push(const DATATYPE & data) {
        std::lock_guard<std::mutex> lg(m_mutex);
        if(m_data.size()>maxSize_)
            return 0;
        m_data.push(data);
        m_cond.notify_one();
        return m_data.size();
    }
    /*
    void push(DATATYPE && data) {
        std::lock_guard<std::mutex> lg(m_mutex);
        m_data.push(std::move(data));
        m_cond.notify_one();
    }*/
    std::shared_ptr<DATATYPE> tryPop() {  // 非阻塞
        std::lock_guard<std::mutex> lg(m_mutex);
        if (m_data.empty()) return {};

        auto res = std::make_shared<DATATYPE>(m_data.front());
        m_data.pop();
        return res;
    }
    
    std::shared_ptr<DATATYPE> pop() {  // 阻塞
        std::unique_lock<std::mutex> lg(m_mutex);
        m_cond.wait(lg, [this] { return !m_data.empty(); });

        auto res = std::make_shared<DATATYPE>(std::move(m_data.front()));
        m_data.pop();
        return res;
    }
    std::shared_ptr<DATATYPE> pop(int sec) {  // 带超时的阻塞
        std::unique_lock<std::mutex> lg(m_mutex);

        bool notempty = m_cond.wait_for(lg, std::chrono::seconds(sec),[this] { return !m_data.empty(); });
        if(!notempty)//if(cs == std::cv_status::timeout || m_data.empty()) 
            return {};

        auto res = std::make_shared<DATATYPE>(std::move(m_data.front()));
        m_data.pop();
        return res;
    }
/*
    std::move唯一的功能是将一个左值强制转化为右值引用，继而可以通过右值引用使用该值，以用于移动语义。
    从实现上讲，std::move基本等同于一个类型转换：static_cast<T&&>(lvalue);
    C++ 标准库使用比如vector::push_back 等这类函数时,会对参数的对象进行复制,连数据也会复制.
    这就会造成对象内存的额外创建, 本来原意是想把参数push_back进去就行了,通过std::move，可以避免不必要的拷贝操作。
*/
private:
    int maxSize_=100000;
    std::queue<DATATYPE, SEQUENCE> m_data;
    mutable std::mutex m_mutex;
    std::condition_variable m_cond;
};
#endif