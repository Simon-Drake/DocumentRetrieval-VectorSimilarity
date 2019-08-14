import math

class Retrieve:
    # Create new Retrieve object storing index and termWeighting scheme
    def __init__(self, index, termWeighting):
        self.index = index
        self.termWeighting = termWeighting
        

        # Calculate full set of documents |D|
        # Set as instance variable so only needs to be calculated once
        fullSetDocs = set()
        for term in self.index:
            docs = set(self.index[term].keys())
            fullSetDocs = fullSetDocs | docs
        self.totalDocs = len(fullSetDocs)

    # Compute cosine similarity
    def similarity(self, query_values, doc_values, candidateDocs):

        cosSimilarity = {}
        # Compute similarity between query and each document
        for document in candidateDocs:

            # If the term weighting is binary all query values = 1
            # No need to use a list comprehension as all squared weights are the same as their base values.
            if self.termWeighting == "binary":
                docLength = math.sqrt(sum(doc_values[document].values()))
                dotProduct = 0
                for term in query_values:
                    if term not in doc_values[document]:
                        continue
                    dotProduct += 1*doc_values[document][term]
                cosSimilarity[document] = dotProduct/docLength
            
            # Where term weighting is not binary use list comprehension and appropriate query term weights.
            else:
                vector = [x*x for x in doc_values[document].values()]
                docLength = math.sqrt(sum(vector))
                dotProduct = 0              
                for term in query_values:
                    if term not in doc_values[document]:
                        continue
                    dotProduct += query_values[term]*doc_values[document][term]
                cosSimilarity[document] = dotProduct/docLength

        # Only take the top 10 (as ir_engine.py would truncate anyway)
        top_10 = sorted(cosSimilarity, key=lambda i:cosSimilarity[i], reverse=True)[:10]
        return top_10

    # Compute idf
    def get_idf(self, term):
        if term in self.index:
            df = len(self.index[term])
            idf = math.log(self.totalDocs/df, 10)
            return idf
        else:
            return 0

    # Computes the tfidf, tf or binary term weighting
    def compute(self, query, candidateDocs):

        doc_values = {}
        query_values = {}

        for term in self.index:

            if self.termWeighting =="tfidf":
                idf = self.get_idf(term)

            # Store the corresponding value for Query
            # Only compute tfidf if needed. 
            # query is already in the format needed for term frequency so we can use that if -w tf
            # For binary term weighting all value for query terms weights are 1 so no need to consider
            if self.termWeighting == "tfidf" and term in query:
                qtf = query[term]
                query_values[term] = qtf*idf

            # Store the corresponding value for each document
            for document in self.index[term]:

                # Create subdictionary if it doesn't exist
                if document not in doc_values:
                    doc_values[document] = {}

                # If the term is not in a document tfidf, tf and b are 0    
                if term not in self.index:
                    doc_values[document][term] = 0
                    continue

                # Retrieve the term frequency for a document
                # If the term is in a document, the weight for binary term weighting is 1
                if document in self.index[term]:
                    dtf = self.index[term][document]
                    if self.termWeighting == "binary":
                        doc_values[document][term] = 1
                        continue
                else:
                    dtf = 0
                # Store appropriate value
                # If term weighting is binary and the term is not in the document dtf = 0 and so the value stored is 0
                if self.termWeighting == "tfidf":
                    doc_values[document][term] = dtf*idf
                else:
                    doc_values[document][term] = dtf

        if self.termWeighting == "tfidf":
            result = self.similarity(query_values, doc_values, candidateDocs)
        else:
            result = self.similarity(query, doc_values, candidateDocs)
        return result

    # Method performing retrieval for specified query
    # Uses helper methods
    def forQuery(self, query):

        # Calculate a set of candidate documents
        candidateDocs = set()
        for term in query:
            if term in self.index:
                docs = set(self.index[term].keys())
                candidateDocs = candidateDocs | docs

        result = self.compute(query, candidateDocs)
        return result