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

#if defined(linux) || defined(__linux) || defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <cctype>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "log.h"

#define EnvToString(envname, dflt) (!getenv(envname) ? (dflt) : getenv(envname))
#define EnvToInt(envname, dflt) (!getenv(envname) ? (dflt) : strtol(getenv(envname), NULL, 10))
#define EnvToBool(envname, dflt) (!getenv(envname) ? (dflt) : memchr("tTyY1\0", getenv(envname)[0], 6) != NULL)

#define GET_ENV_STRING(name, value) do { name = EnvToString("CNINDEX_"#name, value); } while (0)
#define GET_ENV_INT32(name, value) do { name = EnvToInt("CNINDEX_"#name, value); } while (0)
#define GET_ENV_BOOL(name, value) do { name = EnvToBool("CNINDEX_"#name, value); } while (0)

// Based on: https://github.com/google/glog/blob/master/src/utilities.cc
static pid_t GetTID() {
  // On Linux we try to use gettid().
#if defined(linux) || defined(__linux) || defined(__linux__)
#ifndef __NR_gettid
#if !defined __i386__
#error "Must define __NR_gettid for non-x86 platforms"
#else
#define __NR_gettid 224
#endif
#endif
  static bool lacks_gettid = false;
  if (!lacks_gettid) {
    pid_t tid = syscall(__NR_gettid);
    if (tid != -1) {
      return tid;
    }
    // Technically, this variable has to be volatile, but there is a small
    // performance penalty in accessing volatile variables and there should
    // not be any serious adverse effect if a thread does not immediately see
    // the value change to "true".
    lacks_gettid = true;
  }
#endif  // OS_LINUX

  // If gettid() could not be used, we use one of the following.
#if defined(linux) || defined(__linux) || defined(__linux__)
  return getpid();  // Linux:  getpid returns thread ID when gettid is absent
#elif defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
  return GetCurrentThreadId();
#else
  // If none of the techniques above worked, we use pthread_self().
  return (pid_t)(uintptr_t)pthread_self();
#endif
}

/**
 * @brief Remove all spaces in the string
 */
static std::string StringTrim(const std::string& str) {
  std::string::size_type index = 0;
  std::string result = str;

  while ((index = result.find(' ', index)) != std::string::npos) {
    result.erase(index, 1);
  }
  return result;
}

static double GetTimeStamp() {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  double now = (static_cast<int64_t>(ts.tv_sec) * 1000000 + ts.tv_nsec / 1000) * 0.000001;
  return now;
}

// cycle clock is retuning microseconds since the epoch.
static size_t CycleClock_Now() { return static_cast<size_t>(GetTimeStamp() * 1000000); }

static const char* const_basename(const char* filepath) {
  const char* base = strrchr(filepath, '/');
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
  if (!base) base = strrchr(filepath, '\\');
#endif
  return base ? (base + 1) : filepath;
}

namespace cnindex {

constexpr int NUM_SEVERITIES = 8;
const char* const LogSeverityNames[NUM_SEVERITIES] =
    {"CONST", "FATAL", "ERROR", "WARNING", "INFO", "DEBUG", "TRACE", "VERBOSE"};

/**
 * @brief Log filter.
 *
 * Usage: export CNINDEX_LOG_FILTER=IVFPQ:2,IVFSQ:3 ...
 */
static std::string LOG_FILTER = "";
static int LOG_LEVEL = cnindex::LOG_INFO;

enum LogColor { COLOR_DEFAULT, COLOR_RED, COLOR_GREEN, COLOR_YELLOW };

static LogColor SeverityToColor(LogSeverity severity) {
  assert(severity >= 0 && severity < NUM_SEVERITIES);
  LogColor color = COLOR_DEFAULT;
  switch (severity) {
    case LOG_INFO:
    case LOG_DEBUG:
    case LOG_TRACE:
    case LOG_VERBOSE:
    case LOG_CONST:
      color = COLOR_DEFAULT;
      break;
    case LOG_WARNING:
      color = COLOR_YELLOW;
      break;
    case LOG_ERROR:
    case LOG_FATAL:
      color = COLOR_RED;
      break;
    default:
      // should never get here.
      assert(false);
  }
  return color;
}

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
// Returns the character attribute for the given color.
static WORD GetColorAttribute(LogColor color) {
  switch (color) {
    case COLOR_RED:
      return FOREGROUND_RED;
    case COLOR_GREEN:
      return FOREGROUND_GREEN;
    case COLOR_YELLOW:
      return FOREGROUND_RED | FOREGROUND_GREEN;
    default:
      return 0;
  }
}
#else
// Returns the ANSI color code for the given color.
static const char* GetAnsiColorCode(LogColor color) {
  switch (color) {
    case COLOR_RED:
      return "1";
    case COLOR_GREEN:
      return "2";
    case COLOR_YELLOW:
      return "3";
    case COLOR_DEFAULT:
      return "";
  }
  return NULL;  // stop warning about return type.
}
#endif  // OS_WINDOWS

static void ColoredWriteToStderr(LogSeverity severity, const char* message, size_t len) {
  const LogColor color = SeverityToColor(severity);

  // Avoid using cerr from this module since we may get called during
  // exit code, and cerr may be partially or fully destroyed by then.
  if (COLOR_DEFAULT == color) {
    fwrite(message, len, 1, stderr);
    return;
  }
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
  const HANDLE stderr_handle = GetStdHandle(STD_ERROR_HANDLE);

  // Gets the current text color.
  CONSOLE_SCREEN_BUFFER_INFO buffer_info;
  GetConsoleScreenBufferInfo(stderr_handle, &buffer_info);
  const WORD old_color_attrs = buffer_info.wAttributes;

  // We need to flush the stream buffers into the console before each
  // SetConsoleTextAttribute call lest it affect the text that is already
  // printed but has not yet reached the console.
  fflush(stderr);
  SetConsoleTextAttribute(stderr_handle, GetColorAttribute(color) | FOREGROUND_INTENSITY);
  fwrite(message, len, 1, stderr);
  fflush(stderr);
  // Restores the text color.
  SetConsoleTextAttribute(stderr_handle, old_color_attrs);
#else
  fprintf(stderr, "\033[0;3%sm", GetAnsiColorCode(color));
  fwrite(message, len, 1, stderr);
  fprintf(stderr, "\033[m");  // Resets the terminal to default.
#endif  // OS_WINDOWS
}

static void WriteToStderr(const char* message, size_t len) {
  // Avoid using cerr from this module since we may get called during
  // exit code, and cerr may be partially or fully destroyed by then.
  fwrite(message, len, 1, stderr);
}

using CategoryFilterMaps = std::unordered_map<std::string, LogSeverity>;

static std::shared_ptr<CategoryFilterMaps> CreateFilterMaps() {
  std::string filter_str = StringTrim(LOG_FILTER);
  if (filter_str.empty()) {
    return nullptr;
  }

  std::shared_ptr<CategoryFilterMaps> maps = std::make_shared<CategoryFilterMaps>();

  const char* category = filter_str.c_str();
  const char* sep;
  while ((sep = strchr(category, ':')) != NULL) {
    std::string pattern(category, sep - category);
    std::transform(pattern.begin(), pattern.end(), pattern.begin(), [](unsigned char c) { return std::toupper(c); });
    int category_level = LOG_LEVEL;
    if (sscanf(sep, ":%d", &category_level) != 1) {
      std::cout << "Parse " << pattern << " log level failed, will set to " << category_level << std::endl;
    }
    maps->insert(std::make_pair(pattern, LogSeverity(category_level)));
    // Skip past this entry
    category = strchr(sep, ',');
    if (category == nullptr) break;
    category++;  // Skip past ","
  }
  return maps;
}

static bool CategoryActivated(const char* category, LogSeverity severity) {
  static std::shared_ptr<CategoryFilterMaps> filter_maps = CreateFilterMaps();
  severity = severity > LOG_CONST ? severity : LOG_CONST;
  if (filter_maps) {
    auto it = filter_maps->find(std::string(category));
    return it != filter_maps->end() && it->second >= severity;
  }
  return LOG_LEVEL >= severity;
}

const size_t LogMessage::MaxLogMsgLen = 1024;

struct LogMessage::LogMessageData {
  LogMessageData();
  // Buffer space; contains complete message text.
  char message_buf_[LogMessage::MaxLogMsgLen + 1];
  LogStream stream_;
  LogSeverity severity_;     // What level is this LogMessage logged at?
  int line_;                 // line number where logging call is.
  time_t timestamp_;         // Time of creation of LogMessage
  struct ::tm tm_time_;      // Time of creation of LogMessage
  int32_t usecs_;            // Time of creation of LogMessage - microseconds part
  size_t num_prefix_chars_;  // # of chars of prefix in this message
  size_t num_chars_to_log_;  // # of chars of msg to send to log
  const char* filename_;     // basename of file that called LOG
  const char* category_;     // Which category call is.
  bool has_been_flushed_;    // false => data has not been flushed

 private:
  LogMessageData(const LogMessageData&) = delete;
  void operator=(const LogMessageData&) = delete;
};  // struct LogMessageData

static thread_local bool thread_msg_data_available = true;
static thread_local std::aligned_storage<sizeof(LogMessage::LogMessageData), alignof(LogMessage::LogMessageData)>::type
    thread_msg_data;
static std::mutex init_mutex;  // init mutex

LogMessage::LogMessageData::LogMessageData() : stream_(message_buf_, LogMessage::MaxLogMsgLen) {}

LogMessage::LogMessage(const char* category, const char* file, int line, LogSeverity severity) : allocated_(nullptr) {
  Init(category, file, line, severity);
}

LogMessage::~LogMessage() {
  Flush();
  if (data_ == static_cast<void*>(&thread_msg_data)) {
    data_->~LogMessageData();
    thread_msg_data_available = true;
  } else {
    delete allocated_;
  }
}

void LogMessage::Init(const char* category, const char* file, int line, LogSeverity severity) {
  if (thread_msg_data_available) {
    thread_msg_data_available = false;
    data_ = new (&thread_msg_data) LogMessageData();
  } else {
    allocated_ = new LogMessageData();
    data_ = allocated_;
  }

  static bool inited = false;
  if (!inited) {
    std::lock_guard<std::mutex> lk(init_mutex);
    if (!inited) {
      GET_ENV_STRING(LOG_FILTER, "");
      GET_ENV_INT32(LOG_LEVEL, LogSeverity::LOG_INFO);
      inited = true;
    }
  }

  if (!CategoryActivated(category, severity)) {
    data_->has_been_flushed_ = true;
    return;
  }

#ifndef DEBUG
  if (severity == LOG_CONST) {
    data_->num_chars_to_log_ = 0;
    data_->num_prefix_chars_ = 0;
    data_->has_been_flushed_ = false;
    return;
  }
#endif

  stream().fill('0');
  data_->severity_ = severity;
  data_->line_ = line;
  double now = GetTimeStamp();
  data_->timestamp_ = static_cast<time_t>(now);
  localtime_r(&data_->timestamp_, &data_->tm_time_);
  data_->usecs_ = static_cast<int32_t>((now - data_->timestamp_) * 1000000);

  data_->num_chars_to_log_ = 0;
  data_->filename_ = const_basename(file);
  data_->category_ = category;
  data_->has_been_flushed_ = false;

  stream() << "CNINDEX " << data_->category_ << ' '
           << LogSeverityNames[severity][0]
           // << setw(4) << 1900+data_->tm_time_.tm_year
           << std::setw(2) << 1 + data_->tm_time_.tm_mon << std::setw(2) << data_->tm_time_.tm_mday << ' '
           << std::setw(2) << data_->tm_time_.tm_hour << ':' << std::setw(2) << data_->tm_time_.tm_min << ':'
           << std::setw(2) << data_->tm_time_.tm_sec << "." << std::setw(6) << data_->usecs_ << ' ' << std::setfill(' ')
           << std::setw(5) << static_cast<unsigned int>(GetTID())
#ifdef DEBUG
           << ' ' << data_->filename_ << ':' << data_->line_
#endif
           << "] ";
  data_->num_prefix_chars_ = data_->stream_.pcount();
}

std::ostream& LogMessage::stream() { return data_->stream_; }

void LogMessage::Flush() {
  if (data_->has_been_flushed_) return;
  data_->num_chars_to_log_ = data_->stream_.pcount();

  if (data_->message_buf_[data_->num_chars_to_log_ - 1] != '\n') {
    data_->message_buf_[data_->num_chars_to_log_++] = '\n';
  }

  SendToLog();
}

void LogMessage::SendToLog() {
  ColoredWriteToStderr(data_->severity_, data_->message_buf_, data_->num_chars_to_log_);
  if (data_->severity_ == LOG_FATAL) abort();
}

}  // namespace cnindex
