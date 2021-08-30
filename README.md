# thesis-ambiguity_detection
Methods to detect ambiguity in an open-domain conversation
This repository contains all the code used to explore methods for detection of the level of ambiguity in the first query posed by a user in an open-domain conversation with a speech based search interface.
The document graph approach creates a graph based on the similarity between the SBERT embeddings of the documents retrieved by the search service. The concept graph approach derives concepts from those documents and then uses them to create a graph. These graphs are then processed by a Graph Convolutional Network to classify the ambiguity into 4 levels.
