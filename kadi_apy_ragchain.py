class KadiApyRagchain:
    
    def __init__(self, llm, vector_store):
        """
        Initialize the RAGChain with an LLM instance, a vector store
        """
        self.llm = llm
        self.vector_store = vector_store


    def process_query(self, query, chat_history):
        """
        Process a user query, handle history, retrieve contexts, and generate a response.
        """

        
        # Rewrite query
        rewritten_query = self.rewrite_query(query)
        print("Rewritten Query: ",rewritten_query)
        # Predict library usage
        print("Start prediction:")
        code_library_usage_prediction = self.predict_library_usage(query)
        
        # Retrieve contexts
        print("Start retrieving:")

        
        code_contexts = self.retrieve_contexts(rewritten_query, k=3, filter={"usage": code_library_usage_prediction})
        doc_contexts = self.retrieve_contexts(query, k=2, filter={"dataset_category": "kadi_apy_docs"})
     
        # Format contexts
        print("Formatting docs:")
        formatted_doc_contexts = self.format_documents(doc_contexts)
        formatted_code_contexts = self.format_documents(code_contexts)
        
        # Generate response
        print("Start generatin repsonsse:")
        response = self.generate_response(query, chat_history, formatted_doc_contexts, formatted_code_contexts)
        #response = self.generate_response(query, chat_history, formatted_contexts)
        return response
        
    def rewrite_query(self, query):
        """
        Rewrite the user's query to align with the language and structure of the library's methods and documentation.
        """
        rewrite_prompt = (
            f"""You are an intelligent assistant that helps users rewrite their queries.
                The vectorstore consists of the source code and documentation of a Python library, which enables users to 
                programmatically interact with a REST-like API of a software system. The library methods have descriptive 
                docstrings. Your task is to rewrite the query in a way that aligns with the language and structure of the 
                library's methods and documentation, ensuring optimal retrieval of relevant information.

                Guidelines for rewriting the query:
                    1. Identify the main action the user wants to perform (e.g., "Upload a file to a record," "Get users of a group").
                    2. Remove conversational elements like greetings or pleasantries (e.g., "Hello Chatbot", "I need you to help me with").
                    3. Exclude specific variable values (e.g., "ID of my record is '31'") unless essential to the intent.
                    4. Rephrase the query to match the format and keywords used in the docstrings, focusing on verbs and objects relevant to the action 
                       (e.g., "Add a record to a collection").
                    5. Given the query the user might need more than one action to achieve his goal. In this case the rewritten query has more than one action. 

                    Examples:
                        - User query: "Create a Python script with a method that facilitates the creation of records. This method should accept an array of identifiers 
                          as a parameter and allow metadata to be added to each record."
                        - Rewritten query: "create records, add metadata to record"
                        - User query: "Hi, can you help me write Python code to add a record to a collection? The record ID is '45', and the collection ID is '12'."
                          Rewritten query: "add a record to a collection"
                        - User query: I need a python script with which i create a new record with the title: "Hello World"  and then link the record to a given collection.
                          Rewritten query: "create a new record with title" , "link a record to a collection"

                    Based on these examples and guidelines, rewrite the following user query to align more effectively with the keywords used in the docstrings. 
                    Do not include any addition comments, explanations, or text.
                    
                    Original query:
                    {query}
            """
        )
        return self.llm.invoke(rewrite_prompt).content
    
    def predict_library_usage(self, query):
        """
        Use the LLM to predict the relevant library for the user's query.
        """
        prompt = (
            f"""The query is: '{query}'.
                Based on the user's query, assist them by determining which technical document they should read to interact with the software named 'Kadi4Mat'. 
                There are two different technical documents to choose from:
                    - Document 1: Provides information on how to use a Python library to interact with the HTTP API of 'Kadi4Mat'.
                    - Document 2: Provides information on how to use a Python library to implement custom CLI commands to interact with 'Kadi4Mat'.
        
                Your task is to select the single most likely option. 
                    If Document 1 is the best choice, respond with 'kadi_apy/lib/'. 
                    If Document 2 is the best choice, respond with 'kadi_apy/cli/'. 
                Respond with only the exact corresponding option and do not include any additional comments, explanations, or text."
            """
        )
        return self.llm.predict(prompt)

    def retrieve_contexts(self, query, k, filter = None):
        """
        Retrieve relevant documents and source code based on the query and library usage prediction.
        """
        context = self.vector_store.similarity_search(query = query, k=k, filter=filter)       
        return context

    def generate_response(self, query, chat_history, doc_context, code_context):
        """
        Generate a response using the retrieved contexts and the LLM.
        """     
        formatted_history = self.format_history(chat_history)
        
        # Update the prompt with history included
        prompt = f"""
            You are a Python programming assistant specialized in the "Kadi-APY" library. 
            The "Kadi-APY" library is a Python package designed to facilitate interaction with the REST-like API of a software platform called Kadi4Mat. 
            Your task is to answer the user's query based on the guidelines, and if needed, combine understanding provided by 
            "Document Snippets" with the implementation details provided by "Code Snippets." 
    
            Guidelines if generating code:
                - Display the complete code first, followed by a concise explanation in no more than 5 sentences.  
                
            General Guidelines:
                - Refer to the "Chat History" if it provides context that could enhance your understanding of the user's query.
                - Always include the "Chat History" if relevant to the user's query for continuity and clarity in responses.
                - If the user's query cannot be fulfilled based on the provided snippets, reply with "The API does not support the requested functionality."
                - If the user's query does not implicate any task, reply with a question asking the user to elaborate.
            
            Chat History:
            {formatted_history}
    
            Document Snippets:
            {doc_context}
    
            Code Snippets:
            {code_context}
        
            Query:
            {query}
        """
        return self.llm.invoke(prompt).content

    
    def format_documents(self, documents):
        formatted_docs = []
        for i, doc in enumerate(documents, start=1):
            formatted_docs.append(f"Snippet {i}: \n")            
            formatted_docs.append("\n")
            all_metadata = doc.metadata
            
            metadata_str = ", ".join(f"{key}: {value}" for key, value in all_metadata.items())
            print("\n")
            print("------------------------------Beneath is retrieved doc------------------------------------------------")
            print(metadata_str)
            formatted_docs.append(metadata_str)
            print("\n")
            formatted_docs.append("\n")
            formatted_docs.append(doc.page_content)
            print(doc.page_content)
            print("\n\n")
            print("------------------------------End  of retrived doc------------------------------------------------")
            formatted_docs.append("\n\n")
            
        return formatted_docs

    def format_history(self, chat_history):
        formatted_history = []
        for i, entry in enumerate(chat_history, start=1):
            # Unpack the tuple
            user_query = entry[0] if entry[0] is not None else "No query provided"
            assistant_response = entry[1] if entry[1] is not None else "No response yet"
            
            # Format the history
            formatted_history.append(f"Turn {i}:")
            formatted_history.append(f"User Query: {user_query}")
            formatted_history.append(f"Assistant Response: {assistant_response}")
            formatted_history.append("\n")
        
        return "\n".join(formatted_history)