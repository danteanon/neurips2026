import logging
import os

def setup_logging():
    """
    Set up logging configuration for the entire application.
    """
    # Get the logging level from the environment variable
    log_level = os.environ.get('LOGLEVEL', 'INFO')
    
    # Configure the root logger
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # You can also configure specific loggers differently if needed
    # For example:
    # data_loader_logger = logging.getLogger('data_loaders')
    # data_loader_logger.setLevel(logging.DEBUG)
    
    return log_level