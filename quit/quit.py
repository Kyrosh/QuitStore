#!/usr/bin/env python3

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')))

import argparse
from os.path import join
from quit.conf import QuitConfiguration
from quit.exceptions import InvalidConfigurationError
from quit.web.app import create_app
import handleexit
import logging
from flask import request, Response
from flask.ext.api import FlaskAPI, status
from flask.ext.api.decorators import set_parsers
from flask.ext.api.exceptions import NotAcceptable
from flask.ext.cors import CORS
from rdflib import ConjunctiveGraph, Graph, Literal
import json
import subprocess

werkzeugLogger = logging.getLogger('werkzeug')
werkzeugLogger.setLevel(logging.INFO)

logger = logging.getLogger('core')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('quit.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)


def initialize(args):
    """Build all needed objects.

    Returns:
        A dictionary containing the store object and git repo object.

    """
    gc = False

    if args.disableversioning:
        logger.info('Versioning is disabled')
        v = False
    else:
        logger.info('Versioning is enabled')
        v = True

        if args.garbagecollection:
            try:
                gcAutoThreshold = subprocess.check_output(["git", "config", "gc.auto"]).decode("UTF-8").strip()
                if not gcAutoThreshold:
                    gcAutoThreshold = 256
                    subprocess.check_output(["git", "config", "gc.auto", str(gcAutoThreshold)])
                    logger.info('Set default gc.auto threshold ' + str(gcAutoThreshold))
                gc = True
                logger.info('Garbage Collection is enabled with gc.auto threshold ' + str(gcAutoThreshold))
            except Exception as e:
                """
                Disable garbage collection for the rest of the run because it is likely that git is not available
                """
                gc = False
                logger.info('Git garbage collection could not be configured and was disabled')
                logger.debug(e)

    config = QuitConfiguration(
        versioning=v,
        gc=gc,
        configfile=args.configfile,
        targetdir=args.targetdir,
        repository=args.repourl,
        configmode=args.configmode,
    )

    try:
        gitrepo = GitRepo(
            path=config.getRepoPath(),
            origin=config.getOrigin()
        )
    except Exception as e:
        raise InvalidConfigurationError(e)

    # since repo is handled, we can add graphs to config
    config.initgraphconfig()

    store = MemoryStore()

    # Load data to store
    files = config.getfiles()
    for filename in files:
        filepath = join(config.getRepoPath(), filename)
        graphs = config.getgraphuriforfile(filename)
        graphstring = ''

        for graph in graphs:
            graphstring+= str(graph)

        try:
            store.addfile(filepath, config.getserializationoffile(filename))
            logger.info('Success: Graph with URI: ' + graphstring + ' added to my known graphs list')
        except:
            logger.info('Error: Graph with URI: ' + graphstring + ' not added')
            pass

    # Save file objects per file
    filereferences = {}

    for file in config.getfiles():
        graphs = config.getgraphuriforfile(file)
        content = []
        for graph in graphs:
            content+= store.getgraphcontent(graph)
        fileobject = FileReference(join(config.getRepoPath(), file))
        # TODO: Quick Fix, add sorting to FileReference
        fileobject.setcontent(sorted(content))
        filereferences[file] = fileobject

    logger.info('QuitStore successfully running.')
    logger.info('Known graphs: ' + str(config.getgraphs()))
    logger.info('Known files: ' + str(config.getfiles()))
    logger.debug('Path of Gitrepo: ' + config.getRepoPath())
    logger.debug('Config mode: ' + str(config.getConfigMode()))
    logger.debug('All RDF files found in Gitepo:' + str(config.getgraphsfromdir()))

    return {'store': store, 'config': config, 'gitrepo': gitrepo, 'references': filereferences}

def savedexit():
    """Perform actions to be exevuted on API shutdown.

    Add methods you want to call on unexpected shutdown.
    """
    logger.info("Exiting store")
    store.exit()
    logger.info("Store exited")

    return

def main(config):
    """Start the app."""
    app = create_app(config)
    app.run(debug=True, use_reloader=False)


if __name__ == '__main__':
    graphhelp = """This option tells QuitStore how to map graph files and named graph URIs:
                "localconfig" - Use the given local file for graph settings.
                "repoconfig" - Use the configuration of the git repository for graphs settings.
                "graphfiles" - Use *.graph-files for each RDF file to get the named graph URI."""
    confighelp = """Path of config file (turtle). Defaults to ./config.ttl."""

    parser = argparse.ArgumentParser()
    parser.add_argument('-nv', '--disableversioning', action='store_true')
    parser.add_argument('-gc', '--garbagecollection', action='store_true')
    parser.add_argument('-c', '--configfile', type=str, default='config.ttl', help=confighelp)
    parser.add_argument('-r', '--repourl', type=str, help='A link/URI to a remote repository.')
    parser.add_argument('-t', '--targetdir', type=str, help='The directory of the local store repository.')
    parser.add_argument('-cm', '--configmode', type=str, choices=[
        'graphfiles',
        'localconfig',
        'repoconfig'
    ], help=graphhelp)
    args = parser.parse_args()

    objects = initialize(args)
    store = objects['store']
    config = objects['config']
    gitrepo = objects['gitrepo']
    references = objects['references']
    sys.setrecursionlimit(2 ** 15)

    # The app is started with an exit handler
    with handleexit.handle_exit(savedexit):
        main(config)
