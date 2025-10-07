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
"
!pip install --upgrade pip --root-user-action=ignore
!pip install langchain < 0.4 --quiet --root-user-action=ignore
!pip install --upgrade langchain_community < 0.4 --quiet --root-user-action=ignore
!pip install --upgrade langchain-aws <0.3 --quiet --root-user-action=ignore
!pip install faiss-cpu --quiet --root-user-action=ignore
"











