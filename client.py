#!/usr/bin/env python
# client.py

import xmlrpclib
import os, sys

PORT = 8000   # default port

assert len(sys.argv) > 1

modelFile = sys.argv[1]
print "\nInput file: %s\n" %modelFile

if len(sys.argv) > 2:
   PORT = sys.argv[2]

server = xmlrpclib.ServerProxy('http://127.0.0.1:%s'%PORT)
txt = open(modelFile).read()

print server.tagger.runTagger(txt)
