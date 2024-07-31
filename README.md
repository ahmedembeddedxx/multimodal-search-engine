# Multimodal Search Engine

This project is a Multi-Modal Search Engine developed using CLIP by OpenAI, with Flask API for backend and HTML/CSS for the frontend web application.

## Introduction

This project provides a seamless web interface where users can input text queries, and the system retrieves relevant images based on the textual description based on CLIP architecture
[read the paper.](https://arxiv.org/pdf/2103.00020.pdf)


## Take a look
<img src="https://i.ibb.co/5X0P5kt/Screenshot-2024-04-10-at-11-02-46-PM.jpg" alt="Screenshot-2024-04-10-at-11-02-46-PM" border="0">
<img src="https://i.ibb.co/mC0rZZq/Screenshot-2024-04-10-at-11-03-23-PM.jpg" alt="Screenshot-2024-04-10-at-11-03-23-PM" border="0">
<img src="https://i.ibb.co/PtTgF57/Screenshot-2024-04-10-at-11-03-51-PM.jpg" alt="Screenshot-2024-04-10-at-11-03-51-PM" border="0">
<img src="https://i.ibb.co/yY1cR0q/Screenshot-2024-04-10-at-11-04-14-PM.jpg" alt="Screenshot-2024-04-10-at-11-04-14-PM" border="0">

## Demo Video
[![Watch the YouTube video](https://img.youtube.com/vi/FbiKR7LwRJ0/0.jpg)](https://youtu.be/FbiKR7LwRJ0)

- This video demonstrates how to use our project's main feature. 


## How to use for your own images?
- Sample data of 130 images is present in the file
or
- [See the video](https://youtu.be/gJOLHB6QaO0)
or
- Place your images in ```src/minidata```
- Run the notebook ```src/image-processor```
- Move the data in ```src/image_embeddings``` & the data in ```src/minidata``` to ```flaskapp/image_embeddings``` & ```flaskapp/static``` respectively (caution: transfer the data, not the directories)

## Features

- **Multi-Modal Search:** Users can input textual descriptions of images to retrieve relevant images.
- **Intuitive Web Interface:** The frontend is built using React to provide a user-friendly experience.
- **Scalable Backend:** Flask API serves as the backend, handling requests and interacting with the CLIP model.


Clone the repository:

   ```bash
   git clone https://github.com/ahmedembeddedxx/multimodal-search-engine.git
   ```


## Usage

Start the backend server:

   ```bash
   cd flaskapp/
   flask run
   ```

Access the web application in your browser at `http://127.0.0.1:5000/`.

## Stacks
- [OpenAI](https://openai.com) for developing CLIP.
- [Flask](https://flask.palletsprojects.com/) for the backend framework.


## Future Expectences
- Shift the app to ReactJs
- Use ImageBind by MetaAI
- More accurate modal evaluation
- Integrate Audio & Video Functionality
