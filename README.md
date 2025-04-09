# TextRecognition

<img align="right" height="300px" width="auto" src="https://github.com/Davigbit/TextRecognition/blob/main/text-recognition-ui/assets/Logo.png">

### Team
- Ahmad Al-Jabi: [LinkedIn](https://www.linkedin.com/in/ahmad-al-jabi/), [GitHub](https://github.com/AhmadAl-Jabi)
- Davi Gava Bittencourt: [LinkedIn](https://www.linkedin.com/in/davigbit/), [Github](https://github.com/Davigbit)

### Project Overview

TextRecognition is a CNN built during the Winter 2025 edition of MAIS 202. As its name suggest, TextRecognition is able identify all hand-written alphanumerical characters in a word. In terms of dataset, we decided to use a fraction of the NIST dataset complemented by custom data made by us using our own UI.

Dataset: [Google Drive](https://drive.google.com/file/d/14M4CYBoxdYwFgq9y3jQUHPB-riL3e_3C/view?usp=sharing)

### Setting Repo and Installing Dependencies

#### - Clone repo:

```$ git clone https://github.com/Davigbit/TextRecognition```

```$ cd TextRecognition```

#### - Create an environment and install server dependencies on Linux or Mac:

```$ python3 -m venv text-recog-env```

```$ source text-recog-env/bin/activate```

```$ pip install -r requirements.txt```

#### - Create an environment and install server dependencies on Windows:

```$ python -m venv text-recog-env```

```$ text-recog-env/Scripts/activate```

```$ pip install -r requirements.txt```

#### - Install client dependencies:

```$ deactivate```

```$ cd text-recognition-ui```

```$ npm install --legacy-peer-deps```

### Deployement

Open two terminals, one for running the server, the other for the client. 
Also, make sure that both terminals are inside the project's folder.

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

#### - Opening notebooks on Linux or Mac:

```$ source text-recog-env/bin/activate```

```$ jupyter notebook```

#### - Opening notebooks on Windows:

```$ text-recog-env/Scripts/activate```

```$ jupyter notebook```
