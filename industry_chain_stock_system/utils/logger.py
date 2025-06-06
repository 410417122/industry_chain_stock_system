"""
日志管理工具
Logging management utility

提供统一的日志配置和管理功能
Provides unified logging configuration and management functionality
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# 导入第三方库 - Import third-party libraries
try:
    from loguru import logger as loguru_logger
    HAS_LOGURU = True
except ImportError:
    HAS_LOGURU = False

def setup_logger(
    name: str = "industry_chain_system",
    level: str = "INFO",
    log_file: Optional[str] = None,
    use_loguru: bool = True
) -> logging.Logger:
    """
    设置系统日志记录器
    Setup system logger
    
    Args:
        name: 日志记录器名称 - Logger name
        level: 日志级别 - Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径 - Log file path (optional)
        use_loguru: 是否使用loguru库 - Whether to use loguru library
        
    Returns:
        logging.Logger: 配置好的日志记录器 - Configured logger
    """
    
    # 如果指定使用loguru且已安装 - If specified to use loguru and it's installed
    if use_loguru and HAS_LOGURU:
        return setup_loguru_logger(name, level, log_file)
    else:
        return setup_standard_logger(name, level, log_file)

def setup_loguru_logger(
    name: str,
    level: str,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    使用loguru设置高级日志记录器
    Setup advanced logger using loguru
    
    Args:
        name: 日志记录器名称 - Logger name
        level: 日志级别 - Log level
        log_file: 日志文件路径 - Log file path
        
    Returns:
        logging.Logger: loguru兼容的日志记录器 - loguru compatible logger
    """
    
    # 移除默认处理器 - Remove default handler
    loguru_logger.remove()
    
    # 添加控制台输出 - Add console output
    loguru_logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        level=level,
        colorize=True
    )
    
    # 如果指定了日志文件，添加文件输出 - If log file specified, add file output
    if log_file:
        # 确保日志目录存在 - Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 添加文件处理器 - Add file handler
        loguru_logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level=level,
            rotation="10 MB",  # 日志轮转：文件大小达到10MB时轮转 - Log rotation: rotate when file reaches 10MB
            retention="30 days",  # 保留30天的日志 - Retain logs for 30 days
            compression="zip",  # 压缩旧日志文件 - Compress old log files
            encoding="utf-8"
        )
    
    # 创建标准库兼容的适配器 - Create standard library compatible adapter
    class LoguruAdapter:
        """Loguru适配器，提供标准库logging接口 - Loguru adapter providing standard logging interface"""
        
        def __init__(self, logger_name: str):
            self.name = logger_name
            
        def info(self, message: str, *args, **kwargs):
            """记录info级别日志 - Log info level message"""
            loguru_logger.bind(name=self.name).info(message, *args, **kwargs)
            
        def debug(self, message: str, *args, **kwargs):
            """记录debug级别日志 - Log debug level message"""
            loguru_logger.bind(name=self.name).debug(message, *args, **kwargs)
            
        def warning(self, message: str, *args, **kwargs):
            """记录warning级别日志 - Log warning level message"""
            loguru_logger.bind(name=self.name).warning(message, *args, **kwargs)
            
        def error(self, message: str, *args, **kwargs):
            """记录error级别日志 - Log error level message"""
            loguru_logger.bind(name=self.name).error(message, *args, **kwargs)
            
        def critical(self, message: str, *args, **kwargs):
            """记录critical级别日志 - Log critical level message"""
            loguru_logger.bind(name=self.name).critical(message, *args, **kwargs)
            
        def exception(self, message: str, *args, **kwargs):
            """记录异常信息 - Log exception information"""
            loguru_logger.bind(name=self.name).exception(message, *args, **kwargs)
    
    return LoguruAdapter(name)

def setup_standard_logger(
    name: str,
    level: str,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    使用标准库设置日志记录器
    Setup logger using standard library
    
    Args:
        name: 日志记录器名称 - Logger name
        level: 日志级别 - Log level
        log_file: 日志文件路径 - Log file path
        
    Returns:
        logging.Logger: 标准日志记录器 - Standard logger
    """
    
    # 创建日志记录器 - Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 避免重复添加处理器 - Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # 创建格式化器 - Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 添加控制台处理器 - Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, level.upper()))
    logger.addHandler(console_handler)
    
    # 如果指定了日志文件，添加文件处理器 - If log file specified, add file handler
    if log_file:
        # 确保日志目录存在 - Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 添加文件处理器 - Add file handler
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,  # 保留5个备份文件 - Keep 5 backup files
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, level.upper()))
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    获取已配置的日志记录器
    Get configured logger
    
    Args:
        name: 日志记录器名称 - Logger name
        
    Returns:
        logging.Logger: 日志记录器 - Logger
    """
    return logging.getLogger(name)

class LoggerMixin:
    """
    日志记录器混合类
    Logger mixin class
    
    为其他类提供日志记录功能
    Provides logging functionality for other classes
    """
    
    @property
    def logger(self) -> logging.Logger:
        """获取类的日志记录器 - Get logger for the class"""
        if not hasattr(self, '_logger'):
            class_name = self.__class__.__name__
            self._logger = get_logger(f"industry_chain_system.{class_name}")
        return self._logger

# 全局日志记录器实例 - Global logger instance
_system_logger = None

def get_system_logger() -> logging.Logger:
    """
    获取系统全局日志记录器
    Get system global logger
    
    Returns:
        logging.Logger: 系统日志记录器 - System logger
    """
    global _system_logger
    if _system_logger is None:
        _system_logger = setup_logger()
    return _system_logger

# 便捷函数 - Convenience functions
def log_info(message: str, *args, **kwargs):
    """记录info级别日志 - Log info level message"""
    get_system_logger().info(message, *args, **kwargs)

def log_debug(message: str, *args, **kwargs):
    """记录debug级别日志 - Log debug level message"""
    get_system_logger().debug(message, *args, **kwargs)

def log_warning(message: str, *args, **kwargs):
    """记录warning级别日志 - Log warning level message"""
    get_system_logger().warning(message, *args, **kwargs)

def log_error(message: str, *args, **kwargs):
    """记录error级别日志 - Log error level message"""
    get_system_logger().error(message, *args, **kwargs)

def log_exception(message: str, *args, **kwargs):
    """记录异常信息 - Log exception information"""
    get_system_logger().exception(message, *args, **kwargs)

# 日志装饰器 - Log decorators
def log_function_call(func):
    """
    函数调用日志装饰器
    Function call logging decorator
    
    记录函数的调用和返回
    Logs function calls and returns
    """
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_system_logger()
        func_name = f"{func.__module__}.{func.__name__}"
        
        # 记录函数调用 - Log function call
        logger.debug(f"调用函数 {func_name}，参数: args={args}, kwargs={kwargs}")
        
        try:
            # 执行函数 - Execute function
            result = func(*args, **kwargs)
            
            # 记录函数返回 - Log function return
            logger.debug(f"函数 {func_name} 执行成功")
            
            return result
            
        except Exception as e:
            # 记录异常 - Log exception
            logger.error(f"函数 {func_name} 执行失败: {e}")
            raise
    
    return wrapper

def log_execution_time(func):
    """
    函数执行时间日志装饰器
    Function execution time logging decorator
    
    记录函数的执行时间
    Logs function execution time
    """
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_system_logger()
        func_name = f"{func.__module__}.{func.__name__}"
        
        # 记录开始时间 - Record start time
        start_time = time.time()
        logger.debug(f"开始执行函数 {func_name}")
        
        try:
            # 执行函数 - Execute function
            result = func(*args, **kwargs)
            
            # 计算执行时间 - Calculate execution time
            execution_time = time.time() - start_time
            logger.info(f"函数 {func_name} 执行完成，耗时: {execution_time:.2f}秒")
            
            return result
            
        except Exception as e:
            # 计算执行时间 - Calculate execution time
            execution_time = time.time() - start_time
            logger.error(f"函数 {func_name} 执行失败，耗时: {execution_time:.2f}秒，错误: {e}")
            raise
    
    return wrapper

if __name__ == "__main__":
    # 测试日志功能 - Test logging functionality
    logger = setup_logger("test_logger", "DEBUG", "logs/test.log")
    
    logger.info("这是一条信息日志")
    logger.debug("这是一条调试日志")
    logger.warning("这是一条警告日志")
    logger.error("这是一条错误日志")
    
    print("日志测试完成，请检查控制台输出和日志文件")
