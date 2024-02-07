#!/usr/bin/python3
#   about: General auxilliary script: 
#          Logging to a file as well as to stdout 
#  author: Peter Karacsonyi <peter.karacsonyi85@gmail.com>
#    date: 21 Feb 2023
# license: GNU General Public License, version 2
#####

import logging, os


def setLogger(filename):
    """
    logging config : set DEBUG level to see all outputs
    logging to stdout as well as to a file

    """
    
    log_format = logging.Formatter('%(message)s')
    log_file = logging.FileHandler(filename=filename)
    log_file.setFormatter(log_format)
    log = logging.getLogger("spl")
    if os.environ.get('LOGLEVEL') is not None and os.environ.get(
            'LOGLEVEL') == "DEBUG":
        log.setLevel(getattr(logging, 'DEBUG'))
    else:
        log.setLevel(getattr(logging, 'INFO'))
    log.addHandler(log_file)
    log_stream = logging.StreamHandler()
    log_stream.setFormatter(log_format)
    log.addHandler(log_stream)