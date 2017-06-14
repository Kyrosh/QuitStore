import os
import pygit2
import logging
import functools as ft
import re

from datetime import datetime
from os import makedirs
from os.path import abspath, exists, isdir, join
from subprocess import Popen

import pygit2
from pygit2 import GIT_MERGE_ANALYSIS_NORMAL, GIT_MERGE_ANALYSIS_UP_TO_DATE, GIT_MERGE_ANALYSIS_FASTFORWARD
from pygit2 import GIT_SORT_REVERSE, GIT_RESET_HARD, GIT_STATUS_CURRENT
from pygit2 import init_repository, clone_repository
from pygit2 import Repository, Signature, RemoteCallbacks, Keypair, UserPass

from rdflib import Graph, ConjunctiveGraph, Dataset, Literal, URIRef, BNode, Namespace
from rdflib import plugin
from rdflib.store import Store as DefaultStore
from rdflib.plugins.memory import IOMemory
from rdflib.graph import Node, ReadOnlyGraphAggregate, ModificationException, UnSupportedAggregateOperation, Path

from quit.namespace import RDF, RDFS, FOAF, XSD, PROV, QUIT, is_a
from quit.exceptions import RepositoryNotFound, RevisionNotFound, NodeNotFound
from quit.utils import graphdiff, clean_path

from datetime import datetime

corelogger = logging.getLogger('core.quit')

#######################################
# Store implementation
#######################################

class Queryable:
    """
    A class that represents a querable graph-like object.
    """
    def __init__(self, **kwargs):
        self.store = ConjunctiveGraph(identifier='default')

    def query(self, querystring):
        """Execute a SPARQL select query.

        Args:
            querystring: A string containing a SPARQL ask or select query.
        Returns:
            The SPARQL result set
        """
        pass

    def update(self, querystring, versioning=True):
        """Execute a SPARQL update query and update the store.

        This method executes a SPARQL update query and updates and commits all affected files.

        Args:
            querystring: A string containing a SPARQL upate query.
        """
        pass

class Store(Queryable):
    """A class that combines and syncronieses n-quad files and an in-memory quad store.

    This class contains information about all graphs, their corresponding URIs and
    pathes in the file system. For every Graph (context of Quad-Store) exists a
    FileReference object (n-quad) that enables versioning (with git) and persistence.
    """

    def __init__(self, store):
        """Initialize a new MemoryStore instance."""
        self.logger = logging.getLogger('memory_store.core.quit')
        self.logger.debug('Create an instance of MemoryStore')
        self.store = store

        return

    def getgraphuris(self):
        """Method to get all available named graphs.

        Returns:
            A list containing all graph uris found in store.
        """
        graphs = []
        for graph in self.store.contexts():
            if isinstance(graph, BNode) or str(graph.identifier) == 'default':
                pass
            else:
                graphs.append(graph.identifier)

        return graphs

    def getgraphcontent(self, graphuri):
        """Get the serialized content of a named graph.

        Args:
            graphuri: The URI of a named graph.
        Returns:
            content: A list of strings where each string is a quad.
        """
        data = []
        context = self.store.get_context(URIRef(graphuri))
        triplestring = context.serialize(format='nt').decode('UTF-8')

        # Since we have triples here, we transform them to quads by adding the graphuri
        # TODO This might cause problems if ' .\n' will be part of a literal.
        #   Maybe a regex would be a better solution
        triplestring = triplestring.replace(' .\n', ' <' + graphuri + '> .\n')

        data = triplestring.splitlines()
        data.remove('')

        return data

    def getstoreobject(self):
        """Get the conjunctive graph object.

        Returns:
            graph: A list of strings where each string is a quad.
        """
    def graphexists(self, graphuri):
        """Ask if a named graph FileReference object for a named graph URI.

        Args:
            graphuri: A string containing the URI of a named graph

        Returns:
            True or False
        """
        if self.store.get_context(URIRef(graphuri)) is None:
            return False
        else:
            return True

    def addfile(self, filename, serialization):
        """Add a file to the store.

        Args:
            filename: A String for the path to the file.
            serialization: A String containg the RDF format
        Raises:
            ValueError if the given file can't be parsed as nquads.
        """
        try:
            self.store.parse(source=filename, format=serialization)
        except:
            self.logger.debug('Could not import', filename, '.')
            self.logger.debug('Make sure the file exists and contains data in', serialization)
            pass

        return

    def addquads(self, quads):
        """Add quads to the MemoryStore.

        Args:
            quads: Rdflib.quads that should be added to the MemoryStore.
        """
        self.store.addN(quads)
        self.store.commit()

        return

    def query(self, querystring):
        """Execute a SPARQL select query.

        Args:
            querystring: A string containing a SPARQL ask or select query.
        Returns:
            The SPARQL result set
        """
        return self.store.query(querystring)

    def update(self, querystring, versioning=True):
        """Execute a SPARQL update query and update the store.

        This method executes a SPARQL update query and updates and commits all affected files.

        Args:
            querystring: A string containing a SPARQL upate query.
        """
        # methods of rdflib ConjunciveGraph
        if versioning:
            actions = evalUpdate(self.store, querystring)
            self.store.update(querystring)
            return actions
        else:
            self.store.update(querystring)
            return

        return

    def removequads(self, quads):
        """Remove quads from the MemoryStore.

        Args:
            quads: Rdflib.quads that should be removed to the MemoryStore.
        """
        self.store.remove((quads))
        self.store.commit()
        return

    def exit(self):
        """Execute actions on API shutdown."""
        return

class MemoryStore(Store):
    def __init__(self, additional_bindings=list()):
        store = ConjunctiveGraph(identifier='default')
        for prefix, namespace in [('quit', QUIT), ('foaf', FOAF)]:
            store.bind(prefix, namespace)
        for prefix, namespace in additional_bindings:
            store.bind(prefix, namespace)
        super().__init__(store=store)

#######################################
# Graph wrapper
#######################################

class VirtualGraph(Queryable):
    def __init__(self, store):
        if not isinstance(store, InMemoryGraphAggregate):
            raise Exception()
        self.store = store

    def query(self, querystring):
        return self.store.query(querystring)

    def update(self, querystring, versioning=True):
        return self.store.update(querystring)

#######################################
# Quit
#######################################

class Quit(object):
    def __init__(self, config, repository, store):
        self.config = config
        self.repository = repository
        self.store = store

    def sync(self, rebuild = False):
        """ 
        Synchronizes store with repository data.
        """
        if rebuild:
            for c in self.store.contexts():
                self.store.remove((None,None,None), c) 

        def exists(id):
            uri = QUIT['version-' + id]
            for _ in self.store.store.quads((uri, None, None, QUIT.default)):
                return True
            return False

        def traverse(commit, seen):
            commits = []
            merges = []

            while True:
                id = commit.id
                if id in seen:
                    break
                seen.add(id)
                if exists(id):
                    break
                commits.append(commit)
                parents = commit.parents
                if not parents:
                    break
                commit = parents[0]
                if len(parents) > 1:
                    merges.append((len(commits), parents[1:]))
            for idx, parents in reversed(merges):
                for parent in parents:
                    commits[idx:idx] = traverse(parent, seen)
            return commits

        seen = set()

        for name in self.repository.tags_or_branches:
            initial_commit = self.repository.revision(name);            
            commits = traverse(initial_commit, seen)

            prov = self.changesets(commits)                          
            self.store.addquads((s, p, o, c) for s, p, o, c in prov.quads())

            #for commit in commits:
                #(_, g) = commit.__prov__()
                #self.store += g
            

    def instance(self, id=None, sync=False):                
        default_graphs = list()

        if id:
            commit = self.repository.revision(id)
        
            _m = self.config.getgraphurifilemap()
        
            for entity in commit.node().entries(recursive=True):
                # todo check if file was changed
                if entity.is_file:
                    if entity.name not in _m.values():
                        continue

                    tmp = ConjunctiveGraph()
                    tmp.parse(data=entity.content, format='nquads')  

                    for context in (c.identifier for c in tmp.contexts()):                    

                        # Todo: why?
                        if context not in _m:
                            continue

                        identifier = context + '-' + entity.blob.hex
                        rewritten_identifier = context

                        if sync:
                            g = Graph(identifier=rewritten_identifier)
                            g += tmp.triples((None, None, None))
                        else:
                            g = ReadOnlyRewriteGraph(self.store.store.store, identifier, rewritten_identifier)
                        default_graphs.append(g)

        instance = InMemoryGraphAggregate(graphs=default_graphs, identifier='default')    

        return VirtualGraph(instance) 

    def changesets(self, commits=None):
        g = ConjunctiveGraph(identifier=QUIT.default)

        if not commits:
            return g

        last = None

        role_author_uri = QUIT['author']
        role_committer_uri = QUIT['committer']
        
        if commits:            
            g.add((role_author_uri, is_a, PROV['Role']))
            g.add((role_committer_uri, is_a, PROV['Role']))

        while commits:
            commit = commits.pop()
            rev = commit.id

            # Create the commit            
            commit_graph = self.instance(commit.id, True)   
            commit_uri = QUIT['commit-' + commit.id]

            g.add((commit_uri, is_a, PROV['Activity']))

            if 'import' in commit.properties.keys(): 
                g.add((commit_uri, is_a, QUIT['Import']))
                g.add((commit_uri, QUIT['dataSource'], URIRef(commit.properties['import'].strip())))

            g.add((commit_uri, QUIT['hex'], Literal(commit.id)))
            g.add((commit_uri, PROV['startedAtTime'], Literal(commit.author_date, datatype = XSD.dateTime)))
            g.add((commit_uri, PROV['endedAtTime'], Literal(commit.committer_date, datatype = XSD.dateTime)))
            g.add((commit_uri, RDFS['comment'], Literal(commit.message.strip())))

            # Author
            hash = pygit2.hash(commit.author.email).hex
            author_uri = QUIT['user-' + hash]
            g.add((commit_uri, PROV['wasAssociatedWith'], author_uri))
            
            g.add((author_uri, is_a, PROV['Agent']))
            g.add((author_uri, RDFS.label, Literal(commit.author.name)))
            g.add((author_uri, FOAF.mbox, Literal(commit.author.email)))

            q_author_uri = BNode()
            g.add((commit_uri, PROV['qualifiedAssociation'], q_author_uri))
            g.add((q_author_uri, is_a, PROV['Association']))
            g.add((q_author_uri, PROV['agent'], author_uri))
            g.add((q_author_uri, PROV['role'], role_author_uri))

            if commit.author.name != commit.committer.name:
                # Committer
                hash = pygit2.hash(commit.committer.email).hex
                committer_uri = QUIT['user-' + hash]
                g.add((commit_uri, PROV['wasAssociatedWith'], committer_uri))

                g.add((committer_uri, is_a, PROV['Agent']))
                g.add((committer_uri, RDFS.label, Literal(commit.committer.name)))
                g.add((committer_uri, FOAF.mbox, Literal(commit.committer.email)))

                q_committer_uri = BNode()
                g.add((commit_uri, PROV['qualifiedAssociation'], q_committer_uri))
                g.add((q_committer_uri, is_a, PROV['Association']))
                g.add((q_committer_uri, PROV['agent'], author_uri))
                g.add((q_committer_uri, PROV['role'], role_committer_uri))
            else:
                g.add((q_author_uri, PROV['role'], role_committer_uri))

            # Parents
            parent = None
            parent_graph = None

            if commit.parents:
                parent = commit.parents[0]
                parent_graph = self.instance(parent.id, True) 

                for parent in commit.parents:
                    parent_uri = QUIT['commit-' + parent.id]
                    g.add((commit_uri, QUIT["preceedingCommit"], parent_uri))            

            # Diff
            diff = graphdiff(parent_graph.store if parent_graph else None, commit_graph.store if commit_graph else None)
            for ((resource_uri, _), changesets) in diff.items():                
                for (op, update_graph) in changesets: 
                    update_uri = QUIT['update-' + commit.id]
                    op_uri = QUIT[op + '-' + commit.id]
                    g.add((commit_uri, QUIT['updates'], update_uri))
                    g.add((update_uri, QUIT['graph'], resource_uri))
                    g.add((update_uri, QUIT[op], op_uri))                    
                    g.addN((s, p, o, op_uri) for s, p, o in update_graph)

            # Entities
            _m = self.config.getgraphurifilemap()
            
            for entity in commit.node().entries(recursive=True):
                # todo check if file was changed
                if entity.is_file:
                    
                    if entity.name not in _m.values():
                        continue

                    tmp = ConjunctiveGraph()
                    tmp.parse(data=entity.content, format='nquads')  
                    
                    for context in [c.identifier for c in tmp.contexts()]:

                        # Todo: why?
                        if context not in _m:
                            continue

                        public_uri = context
                        private_uri = context + '-' + entity.blob.hex

                        g.add((private_uri, PROV['specializationOf'], public_uri))
                        g.add((private_uri, PROV['wasGeneratedBy'], commit_uri))
                        g.addN((s, p, o, private_uri) for s, p, o in tmp.triples((None, None, None), context))
                                                                       
        return g
   
    def commit(self, graph, message, index, ref, **kwargs):
        if not graph.store.is_dirty:
            return
        
        seen = set()

        index = self.repository.index(index)        

        files = {}

        for context in graph.store.graphs():
            file = self.config.getfileforgraphuri(context.identifier) or self.config.getGlobalFile() or 'unassigned.nq'

            graphs = files.get(file, [])
            graphs.append(context)
            files[file] = graphs

        for file, graphs in files.items():                
            g = ReadOnlyGraphAggregate(graphs)
    
            if len(g) == 0:
                index.remove(file)
            else:
                content = g.serialize(format='nquad-ordered').decode('UTF-8')
                index.add(file, content)

        author = self.repository._repository.default_signature
        id = index.commit(str(message), author.name, author.email, ref=ref)

        if id:            
            self.repository._repository.set_head(id)            
            if not self.repository.is_bare:                            
                self.repository._repository.checkout(ref, strategy=pygit2.GIT_CHECKOUT_FORCE)
            self.sync()

#######################################
# Graph Extensions for rewriting
#######################################

# roles
role_author = QUIT['author']
role_committer = QUIT['committer']

def _git_timestamp(ts, offset):
    import quit.utils as tzinfo
    if offset == 0:
        tz = utc
    else:
        hours, rem = divmod(abs(offset), 60)
        tzname = 'UTC%+03d:%02d' % ((hours, -hours)[offset < 0], rem)
        tz = tzinfo.TZ(offset, tzname)
    return datetime.fromtimestamp(ts, tz)

class Base(object):
    def __init__(self):
        pass

    def __prov__(self):
        pass


class Repository(Base):
    def __init__(self, path, **params):
        origin = params.get('origin', None)

        try:
            self._repository = pygit2.Repository(path)
        except KeyError:
            if not params.get('create', False):
                raise RepositoryNotFound('Repository "%s" does not exist' % path)
                        
            if origin:
                self.callback = self._callback(origin)
                pygit2.clone_repository(url=origin, path=path, bare=False)
            else:
                pygit2.init_repository(path)

        name = os.path.basename(path).lower()

        self.name = name
        self.path = path
        self.params = params

    def _callback(self, origin):
        """Set a pygit callback for user authentication when acting with remotes.
        This method uses the private-public-keypair of the ssh host configuration.
        The keys are expected to be found at ~/.ssh/
        Warning: Create a keypair that will be used only in QuitStore and do not use
        existing keypairs.
        Args:
            username: The git username (mostly) given in the adress.
            passphrase: The passphrase of the private key.
        """
        from os.path import expanduser
        ssh = join(expanduser('~'), '.ssh')
        pubkey = join(ssh, 'id_quit.pub')
        privkey = join(ssh, 'id_quit')

        from re import search
        # regex to match username in web git adresses
        regex = '(\w+:\/\/)?((.+)@)*([\w\d\.]+)(:[\d]+){0,1}\/*(.*)'
        p = search(regex, origin)
        username = p.group(3)

        passphrase = ''

        try:
            credentials = Keypair(username, pubkey, privkey, passphrase)
        except:
            self.logger.debug('GitRepo, setcallback: Something went wrong with Keypair')
            return

        return RemoteCallbacks(credentials=credentials)

    def _clone(self, origin, path):
        try:
            self.addRemote('origin', origin)
            repo = pygit2.clone_repository(url=origin, path=path, bare=False, callbacks=self.callback)
            return repo
        except:
            raise Exception('Could not clone from', origin)

    @property
    def is_empty(self):
        return self._repository.is_empty

    @property
    def is_bare(self):
        return self._repository.is_bare

    def close(self):
        self._repository = None

    def revision(self, id='HEAD'):
        try:
            commit = self._repository.revparse_single(id)
        except KeyError:
            raise RevisionNotFound(id)

        return Revision(self, commit)

    def revisions(self, name=None, order=pygit2.GIT_SORT_REVERSE):
        seen = set()

        def lookup(name):
            for template in ['refs/heads/%s', 'refs/tags/%s']:
                try:                                                           
                    return self._repository.lookup_reference(template % name)
                except KeyError:                                                        
                    pass
            raise RevisionNotFound(ref)
        
        def traverse(ref, seen):
            for commit in self._repository.walk(ref.target, order):
                oid = commit.oid
                if oid not in seen:
                    seen.add(oid)
                    yield Revision(self, commit)

        def iter_commits(name, seen):
            commits = []
            
            if not name:
                for name in self.branches:
                    ref = self._repository.lookup_reference(name)
                    commits += traverse(ref, seen)
            else:
                ref = lookup(name)
                commits += traverse(ref, seen)
            return commits
        
        return iter_commits(name, seen)

    @property
    def branches(self):
        return [x for x in self._repository.listall_references() if x.startswith('refs/heads/')]

    @property
    def tags(self):
        return [x for x in self._repository.listall_references() if x.startswith('refs/tags/')]

    @property
    def tags_or_branches(self):
        return [x for x in self._repository.listall_references() if x.startswith('refs/tags/') or x.startswith('refs/heads/')]

    def index(self, revision=None):
        index = Index(self)
        if revision:
            index.set_revision(revision)
        return index

    def pull(self, remote_name='origin', branch='master'):
        for remote in self._repository.remotes:
            if remote.name == remote_name:
                remote.fetch()
                remote_master_id = self._repository.lookup_reference('refs/remotes/origin/%s' % (branch)).target
                merge_result, _ = self._repository.merge_analysis(remote_master_id)
                
                # Up to date, do nothing
                if merge_result & pygit2.GIT_MERGE_ANALYSIS_UP_TO_DATE:
                    return

                # We can just fastforward
                elif merge_result & pygit2.GIT_MERGE_ANALYSIS_FASTFORWARD:
                    self._repository.checkout_tree(self._repository.get(remote_master_id))
                    try:
                        master_ref = self._repository.lookup_reference('refs/heads/%s' % (branch))
                        master_ref.set_target(remote_master_id)
                    except KeyError:
                        self._repository.create_branch(branch, repo.get(remote_master_id))
                    self._repository.head.set_target(remote_master_id)
                
                elif merge_result & pygit2.GIT_MERGE_ANALYSIS_NORMAL:
                    self._repository.merge(remote_master_id)

                    if self._repository.index.conflicts is not None:
                        for conflict in repo.index.conflicts:
                            print('Conflicts found in:', conflict[0].path)
                        raise AssertionError('Conflicts, ahhhhh!!')

                    user = self._repository.default_signature
                    tree = self._repository.index.write_tree()
                    commit = self._repository.create_commit('HEAD',
                                                user,
                                                user,
                                                'Merge!',
                                                tree,
                                                [self._repository.head.target, remote_master_id])
                    # We need to do this or git CLI will think we are still merging.
                    self._repository.state_cleanup()
                else:
                    raise AssertionError('Unknown merge analysis result')

    def push(self, remote_name='origin', ref='refs/heads/master:refs/heads/master'):
        for remote in self._repository.remotes:
            if remote.name == remote_name:
                remote.push(ref)

    def __prov__(self):

        commit_graph = self.instance(commit.id, True)
        pass


class Revision(Base):
    re_parser = re.compile(
        r'(?P<key>[\w\-_]+): ((?P<value>[^"\n]+)|"(?P<multiline>.+)")',
        re.DOTALL
    )

    def __init__(self, repository, commit):

        message = commit.message.strip()
        properties, message = self._parse_message(commit.message)
        author = Signature(commit.author.name, commit.author.email, _git_timestamp(commit.author.time, commit.author.offset), commit.author.offset)
        committer = Signature(commit.committer.name, commit.committer.email, _git_timestamp(commit.committer.time, commit.committer.offset), commit.committer.offset)

        self.id = commit.hex
        self.short_id = self.id[:10]
        self.message = message
        self.author = author
        self.author_date = author.datetime
        self.committer = committer
        self.committer_date = committer.datetime

        self._repository = repository
        self._commit = commit
        self._parents = None
        self._properties = properties

    def _parse_message(self, message):
        found = dict()
        idx=-1
        lines = message.splitlines()
        for line in lines:
            idx += 1
            m = re.match(self.re_parser, line)
            if m is not None:
                found[m.group('key')] = m.group('value') or m.group('multiline')
            else:
                break
        return (found, '\n'.join(lines[idx:]))

    @property
    def properties(self):
        return self._properties

    @property
    def parents(self):
        if self._parents is None:
            self._parents = [Revision(self._repository, id) for id in self._commit.parents]
        return self._parents

    def node(self, path=None):
        return Node(self._repository, self._commit, path)

    def graph(store):
        mapping = dict()

        for entry in self.node().entries(recursive=True):
            if not entry.is_file:
                continue
            
            for (public_uri, g) in entry.graph(store):
                if public_uri is None:
                    continue

                mapping[public_uri] = g 
            
        return InstanceGraph(mapping) 

    def __prov__(self):

        uri = QUIT['commit-' + self.id]

        g = ConjunctiveGraph(identifier=QUIT.default)

        # default activity
        g.add((uri, is_a, PROV['Activity']))

        # special activity
        if 'import' in self.properties.keys(): 
            g.add((uri, is_a, QUIT['Import']))
            g.add((uri, QUIT['dataSource'], URIRef(self.properties['import'].strip())))

        # properties
        g.add((uri, PROV['startedAtTime'], Literal(self.author_date, datatype = XSD.dateTime)))
        g.add((uri, PROV['endedAtTime'], Literal(self.committer_date, datatype = XSD.dateTime)))
        g.add((uri, RDFS['comment'], Literal(self.message)))

        # parents
        for parent in self.parents:
            parent_uri = QUIT['commit-' + parent.id]
            g.add((uri, QUIT["preceedingCommit"], parent_uri))               

        g.add((role_author, is_a, PROV['Role']))
        g.add((role_committer, is_a, PROV['Role']))

        # author
        (author_uri, author_graph) = self.author.__prov__()

        g += author_graph
        g.add((uri, PROV['wasAssociatedWith'], author_uri))

        qualified_author = BNode()
        g.add((uri, PROV['qualifiedAssociation'], qualified_author))
        g.add((qualified_author, is_a, PROV['Association']))
        g.add((qualified_author, PROV['agent'], author_uri))
        g.add((qualified_author, PROV['role'], role_author))

        # commiter
        if self.author.name != self.committer.name:
            (committer_uri, committer_graph) = self.committer.__prov__()

            g += committer_graph
            g.add((uri, PROV['wasAssociatedWith'], committer_uri))

            qualified_committer = BNode()
            g.add((uri, PROV['qualifiedAssociation'], qualified_committer))
            g.add((qualified_committer, is_a, PROV['Association']))
            g.add((qualified_committer, PROV['agent'], author_uri))
            g.add((qualified_committer, PROV['role'], role_committer))
        else:
            g.add((qualified_author, PROV['role'], role_committer))

        # diff
        diff = graphdiff(parent_graph, commit_graph)
        for ((resource_uri, hex), changesets) in diff.items():
            for (op, update_graph) in changesets:
                update_uri = QUIT['update-' + hex]
                op_uri = QUIT[op + '-' + hex]
                g.add((uri, QUIT['updates'], update_uri))
                g.add((update_uri, QUIT['graph'], resource_uri))
                g.add((update_uri, QUIT[op], op_uri))                    
                g.addN((s, p, o, op_uri) for s, p, o in update_graph)

        # entities
        for entity in self.node().entries(recursive=True):
            for (entity_uri, entity_graph) in self.committer.__prov__():
                g += entity_graph
                g.add((entity_uri, PROV['wasGeneratedBy'], uri))

        return (uri, g)


class Signature(Base):

    def __init__(self, name, email, datetime, offset):
        self.name = name
        self.email = email
        self.offset = offset
        self.datetime = datetime

    def __str__(self):
        return '{name} <{email}> {date}{offset}'.format(**self.__dict__)

    def __repr__(self):
        return '<{0}> {1}'.format(self.__class__.__name__, self.name).encode('UTF-8')

    def __prov__(self):

        hash = pygit2.hash(self.email).hex
        uri = QUIT['user-' + hash]
        
        g = ConjunctiveGraph(identifier=QUIT.default)

        g.add((uri, is_a, PROV['Agent']))
        g.add((uri, RDFS.label, Literal(self.name)))
        g.add((uri, FOAF.mbox, Literal(self.email)))

        return (uri,g)


class Node(Base):
    
    DIRECTORY = "dir"
    FILE = "file"

    def __init__(self, repository, commit, path=None):

        if path in (None, '', '.'):
            self.obj = commit.tree
            self.name = ''
            self.kind = Node.DIRECTORY
            self.tree = self.obj
        else:
            try:
                entry = commit.tree[path]
            except KeyError:
                raise NodeNotFound(path, commit.hex)
            self.obj = repository._repository.get(entry.oid)
            self.name = clean_path(path)
            if self.obj.type == pygit2.GIT_OBJ_TREE:
                self.kind = Node.DIRECTORY
                self.tree = self.obj
            elif self.obj.type == pygit2.GIT_OBJ_BLOB:
                self.kind = Node.FILE
                self.blob = self.obj

        self._repository = repository
        self._commit = commit

    @property
    def is_dir(self) :
        return self.kind == Node.DIRECTORY

    @property
    def is_file(self) :
        return self.kind == Node.FILE

    @property
    def dirname(self):
        return os.path.dirname(self.name)

    @property
    def basename(self):
        return os.path.basename(self.name)

    @property
    def content(self):
        if not self.is_file:
            return None
        return self.blob.data.decode("utf-8")

    def entries(self, recursive=False):
        if isinstance(self.obj, pygit2.Tree):            
            for entry in self.obj:
                dirname = self.is_dir and self.name or self.dirname
                node = Node(self._repository, self._commit, '/'.join(x for x in [dirname, entry.name] if x))

                yield node
                if recursive and node.is_dir and node.obj is not None:
                    for x in node.entries(recursive=True):
                        yield x

    @property
    def content_length(self):
        if self.is_file:
            return self.blob.size
        return None

    def graph(store):
        if self.is_file:

            tmp = ConjunctiveGraph()
            tmp.parse(data=self.content, format='nquads')  

            for context in tmp.context():

                public_uri = QUIT[context]
                private_uri = QUIT[context + '-' + self.blob.hex]
            
                g = ReadOnlyRewriteGraph(entry.blob.hex, identifier=private_uri)
                g.parse(data=entry.content, format='nquads')

                yield (public_uri, g)


    def __prov__(self):       
        if self.is_file:

            tmp = ConjunctiveGraph()
            tmp.parse(data=self.content, format='nquads')  

            for context in tmp.context():
                g = ConjunctiveGraph(identifier=QUIT.default)
                
                public_uri = QUIT[context]
                private_uri = QUIT[context + '-' + self.blob.hex]

                g.add((private_uri, is_a, PROV['Entity']))
                g.add((private_uri, PROV['specializationOf'], public_uri))
                g.addN((s, p, o, private_uri) for s, p, o, _ in tmp.quads(None, None, None, context))

                yield (private_uri, g)

from heapq import heappush, heappop


class Index(object):
    def __init__(self, repository):
        self.repository = repository
        self.revision = None
        self.stash = {}
        self.contents = set()
        self.dirty = False

    def set_revision(self, revision):
        try:
            self.revision = self.repository.revision(revision)
        except RevisionNotFound as e:
            raise IndexError(e)

    def add(self, path, contents, mode=None):
        path = clean_path(path)

        oid = self.repository._repository.create_blob(contents)

        self.stash[path] = (oid, mode or pygit2.GIT_FILEMODE_BLOB)
        self.contents.add(contents)

    def remove(self, path):
        path = clean_path(path)

        self.stash[path] = (None, None)

    def commit(self, message, author_name, author_email, **kwargs):
        if self.dirty:
            raise IndexError('Index already commited')

        ref = kwargs.pop('ref', 'HEAD')
        commiter_name = kwargs.pop('commiter_name', author_name)
        commiter_email = kwargs.pop('commiter_email', author_email)
        parents = kwargs.pop('parents', [self.revision.id] if self.revision else [])

        author = pygit2.Signature(author_name, author_email)
        commiter = pygit2.Signature(commiter_name, commiter_email)

        # Sort index items
        items = sorted(self.stash.items(), key=lambda x: (x[1][0], x[0]))
        print(items)

        # Create tree
        tree = IndexTree(self)
        while len(items) > 0:
            path, (oid, mode) = items.pop(0)

            if oid is None:
                tree.remove(path)
            else:
                tree.add(path, oid, mode)

        oid = tree.write()
        self.dirty = True

        try:
            return self.repository._repository.create_commit(ref, author, commiter, message, oid, parents)
        except Exception as e:
            print(e)
            return None


class IndexHeap(object):
    def __init__(self):
        self._dict = {}
        self._heap = []

    def __len__(self):
        return len(self._dict)

    def get(self, path):
        return self._dict.get(path)

    def __setitem__(self, path, value):
        if path not in self._dict:
            n = -path.count(os.sep) if path else 1
            heappush(self._heap, (n, path))

        self._dict[path] = value

    def popitem(self):
        key = heappop(self._heap)
        path = key[1]
        return path, self._dict.pop(path)


class IndexTree(object):
    def __init__(self, index):
        self.repository = index.repository
        self.revision = index.revision
        self.builders = IndexHeap()
        if self.revision:
            self.builders[''] = (None, self.repository._repository.TreeBuilder(self.revision._commit.tree))
        else:
            self.builders[''] = (None, self.repository._repository.TreeBuilder())

    def get_builder(self, path):
        parts = path.split(os.path.sep)

        # Create builders if needed
        for i in range(len(parts)):
            _path = os.path.join(*parts[0:i + 1])

            if self.builders.get(_path):
                continue

            args = []
            try:
                if self.revision:
                    node = self.revision.node(_path)
                    if node.is_file:
                        raise IndexError('Cannot create a tree builder. "{0}" is a file'.format(node.name))
                    args.append(node.obj.oid)
            except NodeNotFound:
                pass

            self.builders[_path] = (os.path.dirname(
                _path), self.repository._repository.TreeBuilder(*args))

        return self.builders.get(path)[1]

    def add(self, path, oid, mode):
        builder = self.get_builder(os.path.dirname(path))
        builder.insert(os.path.basename(path), oid, mode)

    def remove(self, path):
        self.revision.node(path)
        builder = self.get_builder(os.path.dirname(path))
        builder.remove(os.path.basename(path))

    def write(self):
        """
        Attach and writes all builders and return main builder oid
        """
        # Create trees
        while len(self.builders) > 0:
            path, (parent, builder) = self.builders.popitem()
            if parent is not None:
                oid = builder.write()
                builder.clear()
                self.builders.get(parent)[1].insert(
                    os.path.basename(path), oid, pygit2.GIT_FILEMODE_TREE)

        oid = builder.write()
        builder.clear()

        return oid


#######################################
# Graph Extensions for rewriting
#######################################

class ReadOnlyRewriteGraph(Graph):
    def __init__(self, store='default', identifier = None, rewritten_identifier = None, namespace_manager = None):
        super().__init__(store=store, identifier=rewritten_identifier, namespace_manager=namespace_manager)
        self.__graph = Graph(store=store, identifier=identifier, namespace_manager=namespace_manager)

    def triples(self, triple):
        return self.__graph.triples(triple)
   
    def __cmp__(self, other):
        if other is None:
            return -1
        elif isinstance(other, Graph):
            return -1
        elif isinstance(other, ReadOnlyRewriteGraph):
            return cmp(self.__graph, other.__graph)
        else:
            return -1
    
    def add(self, triple_or_quad):
        raise ModificationException()

    def addN(self, triple_or_quad):
        raise ModificationException()

    def remove(self, triple_or_quad):
        raise ModificationException()

    def __iadd__(self, other):
        raise ModificationException()

    def __isub__(self, other):
        raise ModificationException()

    def parse(self, source, publicID=None, format="xml", **args):
        raise ModificationException()

    def __len__(self):
        return len(self.__graph)


class InMemoryGraphAggregate(ConjunctiveGraph):
    def __init__(self, graphs=list(), identifier=None):                
        self.__memory_store = IOMemory()        
        super().__init__(self.__memory_store, identifier)
        
        assert isinstance(graphs, list), "graphs argument must be a list of Graphs!!"
        self.__graphs = graphs

    class InMemoryGraph(Graph):
        def __init__(self, store = 'default', identifier = None, namespace_manager = None, external = None):            
            super().__init__(store, identifier, namespace_manager)
            self.__external = external

        def force(self):
            if self.__external is not None and self not in self.store.contexts():
                self.store.addN((s, p, o, self) for s, p, o in self.__external.triples((None, None, None)))
        
        def add(self, triple_or_quad):
            self.force()
            super().add(triple_or_quad)

        def addN(self, triple_or_quad):
            self.force()
            super().addN(triple_or_quad)

        def remove(self, triple_or_quad):
            self.force()
            super().remove(triple_or_quad)

    def __repr__(self):
        return "<InMemoryGraphAggregate: %s graphs>" % len(self.graphs)

    @property
    def is_dirty(self):
        return len(self.store) > 0

    def _spoc(self, triple_or_quad, default=False):
        """
        helper method for having methods that support
        either triples or quads
        """
        if triple_or_quad is None:
            return (None, None, None, self.default_context if default else None)
        if len(triple_or_quad) == 3:
            c = self.default_context if default else None
            (s, p, o) = triple_or_quad
        elif len(triple_or_quad) == 4:
            (s, p, o, c) = triple_or_quad
            c = self._graph(c)
        return s,p,o,c

    def _graph(self, c):
        if c is None: return None
        if not isinstance(c, Graph):
            return self.get_context(c)
        else:
            return c

    def add(self, triple_or_quad):
        s,p,o,c = self._spoc(triple_or_quad, default=True)
        self.store.add((s, p, o), context=c, quoted=False)

    def addN(self, quads):
        self.store.addN((s, p, o, self._graph(c)) for s, p, o, c in quads)

    def remove(self, triple_or_quad):
        s,p,o,c = self._spoc(triple_or_quad)
        self.store.remove((s, p, o), context=c_copy(c))

    def contexts(self, triple=None):
        for graph in self.__graphs:
            if graph.identifier not in (c.identifier for c in self.store.contexts()):
                yield graph
        for graph in self.store.contexts():
            yield graph

    graphs = contexts

    def triples(self, triple_or_quad, context=None):
        s,p,o,c = self._spoc(triple_or_quad)
        context = self._graph(context or c)

        if isinstance(p, Path):
            for s, o in p.eval(self, s, o):
                yield s, p, o
        else:
            for graph in self.graphs():
                if context is None or graph.identifier == context.identifier:
                    for s, p, o in graph.triples((s, p, o)):
                        yield s, p, o

    def quads(self, triple_or_quad=None):
        s,p,o,c = self._spoc(triple_or_quad)
        context = self._graph(c)

        for graph in self.graphs():
           if context is None or graph.identifier == context.identifier:
                for s1, p1, o1 in graph.triples((s, p, o)):
                    yield (s1, p1, o1, graph)

    def graph(self, identifier=None):
        for graph in self.graphs():
           if str(graph.identifier) == str(identifier):
               return graph
        
        return self.get_context(identifier)

    def __contains__(self, triple_or_quad):
        (_,_,_,context) = self._spoc(triple_or_quad)
        for graph in self.graphs():
            if context is None or graph.identifier == context.identifier:
                if triple_or_quad[:3] in graph:
                    return True
        return False

    

    def _default(self, identifier):
        return next( (x for x in self.__graphs if x.identifier == identifier), None)

    def get_context(self, identifier, quoted=False):   
        if not isinstance(identifier, Node):
            identifier = URIRef(identifier)     
        return InMemoryGraphAggregate.InMemoryGraph(store=self.__memory_store, identifier=identifier, namespace_manager=self, external=self._default(identifier))

#######################################
# Provenance
#######################################

class Blame(object):
    """
    Reusable Blame object for web client
    """
    def __init__(self, quit):
        self.quit = quit

    def _generate_values(self, quads):
        result = list()

        for quad in quads:  
            (s, p, o, c) = quad

            c.rewrite = True
            
            # Todo: BNodes in VALUES are not supported by specification? Using UNDEF for now
            _s = 'UNDEF' if isinstance(s, BNode) else s.n3()
            _p = 'UNDEF' if isinstance(p, BNode) else p.n3()
            _o = 'UNDEF' if isinstance(o, BNode) else o.n3()
            _c = 'UNDEF' if isinstance(c, BNode) else c.identifier.n3()

            c.rewrite= False

            result.append((_s, _p, _o, _c))
        return result

    def run(self, quads=None, branch_or_ref='master'):
        """
        Annotated every quad with the respective author

        Args:
                querystring: A string containing a SPARQL ask or select query.
        Returns:
                The SPARQL result set
        """


        commit = self.quit.repository.revision(branch_or_ref)
        g = self.quit.instance(branch_or_ref)    

        #if not quads:
        quads = [x for x in g.store.quads((None, None, None))]

        print(quads)

        values = self._generate_values(quads)
        values_string = ft.reduce(lambda acc, quad: acc + '( %s %s %s %s )\n' % quad, values, '') 

        print(values_string)

        q = """
            SELECT ?s ?p ?o ?context ?hex ?name ?email ?date WHERE {                
                ?commit quit:preceedingCommit* ?c .
                ?c      prov:endedAtTime ?date ;
                        prov:qualifiedAssociation ?qa ;
                        quit:updates ?update ;
                        quit:hex ?hex .
                ?qa     prov:agent ?user ;
                        prov:role quit:author .
                ?user   foaf:mbox ?email ;
                        rdfs:label ?name .                    
                ?update quit:graph ?context ;
                        quit:additions ?additions . 
                GRAPH ?additions {
                    ?s ?p ?o 
                } 
                FILTER NOT EXISTS {
                    ?y quit:preceedingCommit+ ?z . 
                    ?z quit:updates ?update2 .
                    ?update2 quit:graph ?g ;
                        quit:removals ?removals . 
                    GRAPH ?removals {
                        ?s ?p ?o 
                    } 
                }
                VALUES (?s ?p ?o ?context) {
                    %s
                }                                 
            }                
            """ % values_string

        return self.quit.store.store.query(q, initNs = { 'foaf': FOAF, 'prov': PROV, 'quit': QUIT }, initBindings = { 'commit': QUIT['commit-' + commit.id] })