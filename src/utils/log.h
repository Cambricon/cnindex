/*************************************************************************
 * Copyright (C) 2021 by Cambricon, Inc. All rights reserved
 *
 * This source code is licensed under the Apache-2.0 license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * A part of this source code is referenced from glog project.
 * https://github.com/google/glog/blob/master/src/logging.cc
 *
 * Copyright (c) 1999, Google Inc.
 *
 * This source code is licensed under the BSD 3-Clause license found in the
 * LICENSE file in the root directory of this source tree.
 *
 *************************************************************************/

#ifndef __CNINDEX_LOG_H__
#define __CNINDEX_LOG_H__

#include <time.h>

#include <ostream>
#include <streambuf>
#include <string>

#define STR(src) #src

#define LOGC(category) cnindex::LogMessage(STR(category), __FILE__, __LINE__, cnindex::LOG_CONST).stream()
#define LOGC_IF(category, condition) !(condition) ? (void)0 : cnindex::LogMessageVoidify() & LOGC(category)

#define LOGF(category) cnindex::LogMessage(STR(category), __FILE__, __LINE__, cnindex::LOG_FATAL).stream()
#define LOGF_IF(category, condition) !(condition) ? (void)0 : cnindex::LogMessageVoidify() & LOGF(category)

#define LOGE(category) cnindex::LogMessage(STR(category), __FILE__, __LINE__, cnindex::LOG_ERROR).stream()
#define LOGE_IF(category, condition) !(condition) ? (void)0 : cnindex::LogMessageVoidify() & LOGE(category)

#define LOGW(category) cnindex::LogMessage(STR(category), __FILE__, __LINE__, cnindex::LOG_WARNING).stream()
#define LOGW_IF(category, condition) !(condition) ? (void)0 : cnindex::LogMessageVoidify() & LOGW(category)

#define LOGI(category) cnindex::LogMessage(STR(category), __FILE__, __LINE__, cnindex::LOG_INFO).stream()
#define LOGI_IF(category, condition) !(condition) ? (void)0 : cnindex::LogMessageVoidify() & LOGI(category)

#define LOGD(category) cnindex::LogMessage(STR(category), __FILE__, __LINE__, cnindex::LOG_DEBUG).stream()
#define LOGD_IF(category, condition) !(condition) ? (void)0 : cnindex::LogMessageVoidify() & LOGD(category)

#define LOGT(category) cnindex::LogMessage(STR(category), __FILE__, __LINE__, cnindex::LOG_TRACE).stream()
#define LOGT_IF(category, condition) !(condition) ? (void)0 : cnindex::LogMessageVoidify() & LOGT(category)

#define LOGV(category) cnindex::LogMessage(STR(category), __FILE__, __LINE__, cnindex::LOG_VERBOSE).stream()
#define LOGV_IF(category, condition) !(condition) ? (void)0 : cnindex::LogMessageVoidify() & LOGV(category)

namespace cnindex {

/**
 * @brief log severity
 * 0: CONST
 * 1: FATAL
 * 2: ERROR
 * 3: WARNING
 * 4: INFO
 * 5: DEBUG
 * 6: TRACE
 * 7: VERBOSE
 */
enum LogSeverity {LOG_CONST = 0, LOG_FATAL, LOG_ERROR, LOG_WARNING, LOG_INFO, LOG_DEBUG, LOG_TRACE, LOG_VERBOSE };

class LogMessageVoidify {
 public:
  LogMessageVoidify() {}
  // This has to be an operator with a precedence lower than << but
  // higher than ?:
  void operator&(std::ostream&) {}
};

class LogMessage {
 public:
  class LogStreamBuf : public std::streambuf {
   public:
    // REQUIREMENTS: "len" must be >= 2 to account for the '\n' and '\0'.
    LogStreamBuf(char* buf, int len) { setp(buf, buf + len - 2); }

    // This effectively ignores overflow.
    virtual int_type overflow(int_type ch) { return ch; }

    // Legacy public ostrstream method.
    size_t pcount() const { return pptr() - pbase(); }
    char* pbase() const { return std::streambuf::pbase(); }
  };  // class LogStreamBuf

  class LogStream : public std::ostream {
   public:
    LogStream(char* buf, int len) : std::ostream(NULL), streambuf_(buf, len) { rdbuf(&streambuf_); }

    // Legacy std::streambuf methods.
    size_t pcount() const { return streambuf_.pcount(); }
    char* pbase() const { return streambuf_.pbase(); }
    char* str() const { return pbase(); }

   private:
    LogStream(const LogStream&) = delete;
    LogStream& operator=(const LogStream&) = delete;
    LogStreamBuf streambuf_;
  };  // class LogStream

  LogMessage(const char* category, const char* file, int line, LogSeverity severity);
  ~LogMessage();
  void Init(const char* category, const char* file, int line, LogSeverity severity);
  std::ostream& stream();
  struct LogMessageData;

 private:
  LogMessage(const LogMessage&) = delete;
  LogMessage& operator=(const LogMessage&) = delete;
  void Flush();
  void SendToLog();
  LogMessageData* data_;
  LogMessageData* allocated_;
  static const size_t MaxLogMsgLen;
};  // class LogMessage

}  // namespace cnindex

#endif  // __CNINDEX_LOG_H__
