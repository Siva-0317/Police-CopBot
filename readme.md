# Police CopBot

Police CopBot is an AI-powered chatbot designed to assist users with police procedures, legal rights, and reporting crimes. This project utilizes machine learning models and a MySQL database to provide accurate responses to user queries.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Setup](#setup)
- [Running the Application](#running-the-application)
- [Usage](#usage)
- [Contributing](#contributing)
- [Contributors](#contributors)

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7 or higher
- MySQL Server installed and running
- A suitable IDE or text editor (e.g., VSCode, PyCharm)
- Basic knowledge of Python and SQL

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/Police-CopBot.git
   cd Police-CopBot
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install required packages:**

   Install the necessary Python packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

## Setup

1. **Set up MySQL Database:**

   - Open MySQL Workbench or your preferred MySQL client.
   - Create a new database named `tpk`.
   - Create a table named `qa_pairs` with the following structure:

     ```sql
     CREATE TABLE qa_pairs (
         id INT AUTO_INCREMENT PRIMARY KEY,
         question TEXT NOT NULL,
         answer TEXT NOT NULL
     );
     ```

2. **Load initial data:**

   - Place your `police_qa_dataset.json` file in the project directory.
   - Run the `p_mysql.py` script to load the initial data into the database:

     ```bash
     python p_mysql.py
     ```

3. **Configure API Keys:**

   - Replace `"api-key"` in `app.py` and `offline_model.py` with your actual Gemini API key.

4. **Model and Embeddings:**

   - Run the `embeddings.py` script to generate and save question embeddings:

     ```bash
     python embeddings.py
     ```

## Running the Application

1. **Run the Flask Application:**

   To start the web application, run:

   ```bash
   python app.py
   ```

   The application will be accessible at `http://127.0.0.1:5000/chat`.

2. **Run the Offline Model:**

   If you want to run the offline version with a GUI, execute:

   ```bash
   python offline_model.py
   ```

   This will open a Tkinter window for interaction.

## Usage

- For the web application, send a POST request to the `/chat` endpoint with a JSON body containing the user message:

  ```json
  {
      "message": "How do I report a stolen vehicle?"
  }
  ```

- For the offline model, type your queries in the GUI and press "Send" to receive responses.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a pull request.

## Contributors

The following members were a part of this project:
1. Dhana Anjana
2. Jamunasree
```

### Notes:
- Replace `https://github.com/yourusername/Police-CopBot.git` with the actual URL of your repository.
- Ensure that the instructions are clear and that any necessary files (like `police_qa_dataset.json`) are included in the repository or specified for download.
