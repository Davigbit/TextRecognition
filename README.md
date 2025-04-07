# TextRecognition

**MAIS 202 Winter 2025**

### Setting things up:

#### - Clone repo:

```$ git clone https://github.com/Davigbit/TextRecognition```

```$ cd TextRecognition```

#### - Create an environment and install dependencies on Linux or Mac:

```$ python3 -m venv text-recog-env```

```$ source text-recog-env/bin/activate```

```$ pip install -r requirements.txt```

#### - Create an environment and install dependencies on Windows:

```$ python -m venv text-recog-env```

```$ text-recog-env/Scripts/activate```

```$ pip install -r requirements.txt```

#### - Install client dependencies:

```$ deactivate```

```$ cd text-recognition-ui```

```$ npm install --legacy-peer-deps```

### Running the project:

Open two terminals, one for running the server, the other for the client. 
Also, make sure to put both terminals in the project folder.

#### - Running server on Linux or Mac:

 ```$ source text-recog-env/bin/activate```

 ```$ python3 src/server.py```

#### - Running server on Windows:

```$ text-recog-env/Scripts/activate```

```$ python src/server.py```

#### - Running client:

```$ cd text-recognition-ui```

```$ npm run dev```

Go to http://localhost:5173/

#### - Opening notebook on Linux or Mac:

```$ source text-recog-env/bin/activate```

```$ jupyter notebook```

#### - Opening notebook on Windows:

```$ text-recog-env/Scripts/activate```

```$ jupyter notebook```
