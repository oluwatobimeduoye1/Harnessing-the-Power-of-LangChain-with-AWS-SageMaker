This project integrates LangChain, a framework for developing applications powered by large language models (LLMs), with AWS SageMaker JumpStart, to deploy and manage foundation models as scalable inference endpoints. The workflow enables developers to use LangChain’s high-level abstractions (prompt templates, chains, and memory) with SageMaker-hosted LLMs, creating end-to-end generative AI solutions — including Q&A systems, RAG applications, code generators, and chatbots.

## Business Benefits
1. Streamlined AI Model Deployment

Reduces the time and complexity of deploying and scaling LLMs.

SageMaker JumpStart provides pre-trained models that can be quickly deployed as endpoints, eliminating the need to build models from scratch.

2. Operational Efficiency

The integration with LangChain simplifies interactions with LLMs using reusable components.

Enables developers and data scientists to focus on use case logic rather than infrastructure management.

3. Scalable, Cost-Optimized AI Solutions

Models are hosted on SageMaker endpoints, which auto-scale to meet demand.

Businesses can start small with lightweight models (e.g., BGE Small) and scale up as data and traffic grow.

4. Customizable and Extensible Applications

Using LangChain, developers can build custom logic for summarization, code generation, and retrieval-augmented generation (RAG).

Applications such as AI chatbots, content summarizers, and document Q&A tools can be rapidly prototyped and deployed.

5. Integration Across AWS Ecosystem

Connects seamlessly with S3, Lambda, API Gateway, and CloudFront, enabling full-stack AI solutions (from backend inference to web APIs and frontend delivery).

Facilitates serverless deployment of AI apps accessible through web interfaces.

6. Enhanced Productivity for Teams

Simplifies collaboration between data scientists, developers, and DevOps engineers.

Reusable code templates and notebooks speed up experimentation and iteration.

This solution hamesses LangChain for streamlining core generative Al techniques, such as embeddings and text generation.
AWS tools, particularly Amazon
SageMaker JumpStart, are used to deploy models on SageMaker endpoints, ensuring smooth integration for real-world applications.

- The process starts with a developer or user accessing SageMaker JumpStart.
- Developers explore and select a suitable prebuilt large language model (LLM) or other model from the available options in SageMaker JumpStart.
- The selected LLM is deployed as a SageMaker endpoint, which provides a scalable web service for making inference requests to the model.
- An Amazon Simple Storage Service (Amazon S3) bucket is used to store data, model artifacts, and other resources related to the LLM deployment and interaction process.
- Developers set up a Jupyter notebook in SageMaker, which serves as the development environment for interacting with the deployed LLM.
- Inside the Jupyter notebook, developers configure LangChain, a toolkit designed to streamline and enhance the interaction with LLMs, including the LLM deployed by SageMaker.
- Through the Jupyter notebook and LangChain, developers can send requests, process data, generate text, and perform various interactions with the deployed LLM. LangChain provides an abstraction layer over direct API calls, making the process more intuitive and efficient.
- Based on the outputs received from the LLM, developers can iteratively refine the interactions, adjust parameters, prompts, or data inputs to achieve better results, using Lang Chain's capabilities.
- Finally, developers can integrate the desired LLM outputs and interactions into broader applications or services, automating processes, enhancing user experiences, or developing new Al-powered features that use the LLM's capabilities.


## Technical Highlights

- AWS SageMaker JumpStart: Model selection and deployment.

- Amazon S3: Storage for datasets, model artifacts, and notebooks.

- LangChain: LLM orchestration, chaining, memory, and prompt engineering.

- Retrieval-Augmented Generation (RAG): Enhances context-aware Q&A.

- AWS Lambda + API Gateway + CloudFront: Enables web API and frontend integration.

- JupyterLab in SageMaker Studio: Development and testing environment.

<img width="2650" height="1621" alt="image" src="https://github.com/user-attachments/assets/7cd0bd2b-077f-48b9-942a-f713d47b2690" />

## Outcome

This project demonstrates a production-ready AI workflow, connecting cloud infrastructure with generative AI models.
It provides a blueprint for:

- Building RAG-based applications with enterprise-level scalability.

- Deploying serverless AI APIs with global reach.

- Accelerating AI-powered software delivery pipelines in the cloud.

# Concept
In this project, we will:
-   Use Amazon SageMaker JumpStart to deploy a foundation model as a hosted inference.
-   Use a SageMaker Studio notebook to investigate the use of a LangChain library and foundation models.
-   Test the different generative Al-based solutions.
-   Review the sample application and test the usage.


## Steps
1. Create S3 bucket
2. Deploy LLM using SageMaker JumpStart
3. Set up LangChain and connect to endpoint
4. Build and test Q&A, RAG, and chatbot apps
5. Integrate with Lambda, API Gateway, CloudFront


# STEP 1
1. Create an S3 bucket 
2. S3 bucket will contain the project codes required for the project. 

# STEP 2
3. Navigate to Amazon sage maker and then open studio. 
4. Go to Sagemaker Jumpstart

Amazon SageMaker JumpStart is an ML hub that can help you accelerate your ML journey. With SageMaker JumpStart, you can evaluate, compare, and select foundation models (FMs) quickly based on pre-defined quality and responsibility metrics to perform tasks such as article summarization and image generation.

5. Providers search box, type:   bge small, Under Models, choose the BGE Small En V1.5 model.
6. For Instance type, review the default value, mLg5.2xlarge.
7. Click Deploy. - Review and confirm the endpoint status is creating.
8. Scroll to the top of the left navigation pane.
9.    In the Applications pane, click JupyterLab.
10.    On the JupyterLab page, under Status, review to confirm that the status is Stopped.
11.    Under Action click Run. Once running open jupyterlab application 

A JupyterLab space is a private or shared space within SageMaker Studio that manages the storage and compute resources needed to run the JupyterLab application.


Amazon SageMaker Studio Lab provides pre-installed environments for your Studio Lab notebook instances. Using
environments, you can start up a Studio Lab notebook instance with the packages you want to use. This is done by installing packages in the environment and then selecting the environment as a Kernel.

Text generation foundation models can be used for a variety of downstream tasks, including text summarization, text classification, question answering, long-form content gen fation, short-form copywriting, information extraction, and more.

### Deploy LLM using SageMaker JumpStart

Step 1. Deploy text generation model.¶
In this step, we will use the SageMaker Python SDK to deploy the Falcon model for text generation. This permissively licensed (Apache-2.0) open source model is trained on the RefinedWeb dataset.
NOTE: The model might take 10 - 15 minutes to deploy.

<img width="2104" height="784" alt="image" src="https://github.com/user-attachments/assets/f1719593-5a85-4ef1-a3df-0e21b2c0be31" />

Step 2. Install LangChain and other required Python modules. : This code installs Python libraries (such as langchain, langchain_community, faiss-cpu, and ipywidgets).

'''!pip install --upgrade pip --root-user-action=ignore
   !pip install langchain < 0.4 --quiet --root-user-action=ignore
   !pip install --upgrade langchain_community < 0.4 --quiet --root-user-action=ignore
   !pip install --upgrade langchain-aws <0.3 --quiet --root-user-action=ignore
   !pip install faiss-cpu --quiet --root-user-action=ignore '''

Step 3: Import the required Python modules and set up the Amazon SageMaker runtime client.
This code imports Python libraries and initializes the SageMaker session.

''' import boto3, json
from typing import List

session = boto3.Session()
sagemaker_runtime_client = session.client("sagemaker-runtime") 
'''

## Set up LangChain and connect to endpoint

Step 4: Use LangChain with an LLM hosted on a SageMaker endpoint to test a basic Q&A app. 
This code imports LangChain modules required to create a basic Q&A application by using SageMaker endpoints. The code also loads the sample strings by using inbuilt tools : 
'' from typing import Dict
from langchain_core.prompts import PromptTemplate 
from langchain_aws import SagemakerEndpoint
from langchain_aws.llms.sagemaker_endpoint import LLMContentHandler
from langchain.docstore.document import Document 
''

## Return to the JupyterLab browser tab.
In the Step 5 code cell, to update the embedding and instruct endpoint names, paste the two names that you just copied.
Make sure to update the jumpstart-embedding-endpoint and falcon-instruct-endpoint with the actual names of your deployed SageMaker endpoints.

Step 5: Test the Q&A chain with a sample question.¶
This code block initializes the question and answering chain with the SageMaker endpoint and prompt template. 

Step 6: Use Retrieval Augmented Generation (RAG), with LangChain and SageMaker endpoints, to build a basic Q&A app.¶

## Build and test Q&A, RAG, and chatbot apps
Step 8: Apply additional use cases.¶
Using LangChain and an LLM helps you create straightforward custom tools, such as code generators and chatbots.
Step 8.1 : Create a code generator with LangChain and an LLM.¶
The following code snippet outlines the setup for automated code generation using LangChain and an LLM.
* The code imports the necessary components from LangChain and initializes the LLM.
* A prompt template is defined to guide the LLM in writing Python functions based on task descriptions.
* A PromptTemplate instance is created, configuring how the task description is processed.
* An LLMChain instance is then established, combining the prompt and the LLM, ready to generate code.
	•	The system generates code by running the LLMChain with a given task description, demonstrating a streamlined method for converting task descriptions into functional Python code.   Step 8.2: Create a chatbot with LangChain and an LLM.¶
This code snippet demonstrates the process of establishing a conversation chain equipped with memory functionality, using LangChain and an LLM.
* The code imports ConversationBufferMemory and ConversationChain from LangChain. The code then initializes the LLM by using a specific model.
* The code creates a memory buffer designed to store and manage the context of a conversation, making sure past interactions can be recalled and used to inform an ongoing dialogue.
* The final step involves constructing the ConversationChain, which seamlessly integrates both the LLM and the newly created memory buffer.
Through this structured approach, the code facilitates the development of conversational AI interactions capable of retaining and using past dialogue. This capability significantly improves the interactions, making them more coherent and context-sensitive, thus mimicking a more natural and engaging conversational experience.

from langchain.memory import ConversationBufferMemory
from langchain import ConversationChain
from langchain.schema import SystemMessage

parameters = { 
    "max_length": 200, 
    "num_return_sequences": 1, 
    "top_k": 10, 
    "top_p": 0.01, 
    "do_sample": False, 
    "temperature": 0, }

sm_llm = SagemakerEndpoint(
    endpoint_name=instruct_endpoint_name,
    region_name='us-east-1',
    model_kwargs=parameters,
    content_handler=content_handler,
)

# Initialize the LLM (in this case, OpenAI's GPT-3)
llm = sm_llm

# # Initialize the memory
memory = ConversationBufferMemory(memory_key="history", return_messages=False)

# # Add system message
memory.chat_memory.add_message(SystemMessage(content="You are a helpful professional assistant. Respond to the question only."))

# Create the conversation chain with memory
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)   Below the final code cell, review the output to see the "Human" text box 
“
def chat_with_ai():
    print("Hi! I'm an AI assistant. How can I help you today?")
    try:
        while True:
            human_input = input("Human: ").strip()
            if human_input.lower() == 'exit':
                print("Assistant: Goodbye!")
                break  # Exit the loop if the user types 'exit'
            # Process the input through the conversation chain using the run method.
            response = conversation.run(input=human_input)
            # Print the AI's response.
            print(f"Assistant: {response}")
    except KeyboardInterrupt:
        print("\nAssistant: Goodbye!")
“  

## Integrate with Lambda, API Gateway, CloudFront

Navigate to the AWS Lambda console.
1. In the left navigation pane, click Functions.
2. In the Functions section, click application_function.   AWS Lambda is a compute service that helps you run code without provisioning or managing servers.

You can create a web API with an HTTP endpoint for your Lambda function by using Amazon API Gateway. API Gateway provides tools for creating and documenting web APIs that route HTTP requests to Lambda functions.
You can secure access to your API with authentication and authorization controls. Your ve traffic over the internet or can be accessible only within your VPC.   1. Review the Function overview section.
- Amazon API Gateway uses the payload to invoke the Lambda function.
-
- Navigate to the Amazon CloudFront console.
1. In the left navigation pane, click Distributions.
2. In the Distributions section, click the available distribution ID.

## TEST THE APPLICATION 

Return to the web application browser tab.
2. For API Gateway URL, paste the URL that you just copied.
3. For Prompt, type any content that you want to summarize. 





