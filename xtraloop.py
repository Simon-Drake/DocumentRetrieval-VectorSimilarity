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

        query_values = {}
        document_sum_squares = {}
        document_dot_product = {}
        document_lengths = {}
        #doc_values = {}
        cosine = {}

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
                # if document not in doc_values:
                #     doc_values[document] = {}

                if document not in document_sum_squares:
                    document_sum_squares[document] = 0
                    document_dot_product[document] = 0

                # If the term is not in a document tfidf, tf and b are 0    
                # if term not in self.index:
                #     #doc_values[document][term] = 0
                #     document_sum_squares[document] += 0 #can probably remove
                #     continue

                # Retrieve the term frequency for a document
                # If the term is in a document, the weight for binary term weighting is 1
                if document in self.index[term]:
                    dtf = self.index[term][document]
                    if self.termWeighting == "binary":
                        #doc_values[document][term] = 1
                        document_sum_squares[document] += 1
                        continue
                else:
                    dtf = 0
                # Store appropriate value
                # If term weighting is binary and the term is not in the document dtf = 0 and so the value stored is 0
                if self.termWeighting == "tfidf":
                    document_sum_squares[document] += (dtf*idf)*(dtf*idf)
                else:
                    document_sum_squares[document] += (dtf)*(dtf)
                    
                if document in candidateDocs:
                    if term in query and self.termWeighting == "tfidf":  #implement query_values = query
                        document_dot_product[document] += dtf*idf*query_values[term]
                    elif term in query:
                        document_dot_product[document] += dtf*query[term]

                document_lengths[document] = math.sqrt(document_sum_squares[document]) # try with another loop

                if document in candidateDocs:
                    cosine[document] = document_dot_product[document]/document_lengths[document]

        # if self.termWeighting == "tfidf":
        #     result = self.similarity(query_values, doc_values, candidateDocs)
        # else:
        #     result = self.similarity(query, doc_values, candidateDocs)
        result = sorted(cosine, key=lambda i:cosine[i], reverse=True)[:10]
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