# Set the list of text codes
TEXTS_CODES = [0,2,6,10,14,18,
               22,26,30,32,36,
               40,44,50,54,58,
               62,66,70,74,78,
               82,86,90,94,98,
               100,104,108,112]

# Set the list of size codes
SIZE_CODES = [0,6,8,10,12,18,
              20,22,24,36,38,
              40,42,48,50,52,
              66,68,70,72,78,
              80,82,84,94,96,
              98,104,106,108,110]

# Set the list of spatial codes
SPATIAL_CODES = [0,14,16,18,20,
                 22,24,26,28,
                 30,44,46,48,
                 50,52,54,56,
                 74,76,78,80,
                 82,84,86,88,
                 100,102,104,106,
                 108,110,112]

# Set the list of counting codes
COUNTING_CODES = [0,32,34,36,38,
                  40,42,44,46,48,
                  50,52,54,56,58,
                  60,62,64,66,68,
                  70,72,74,76,78,
                  80,82,84,86,88,
                  90]

# Set the list of colors
COLOR_CODES = [0,2,4,6,8,10,12,
               14,16,18,20,22,
               24,26,28,30,32,
               34,36,38,40,42,
               44,46,48,50,52,
               54,56,58,60]

# Set the list of emotion codes
EMOTION_CODES = [0,2,4,6,8,14,16,
                 18,20,32,34,36,
                 38,44,46,48,62,
                 64,66,68,74,76,
                 78,80,92,94,96,
                 100,102,104,106]

# Set the list of skills/rules
SKILLS = ["counting", "color", "spatial", "size", "emotion", "text"]

# Set the list of models
MODELS = ["z_image_turbo",
          "qwen_image",
          "stable_diffusion", 
          "stable_cascade", 
          "kandinsky", 
          "runway_ml", 
          "adobe_firefly", 
          "dalle", 
          "flux"]

VALID_EMOTIONS = ['trust',
                  'disgust', 
                  'anticipation', 
                  'joy', 
                  'surprise', 
                  'fear', 
                  'anger', 
                  'sadness']

VALID_COLORS = ['white',
                'black', 
                'blue', 
                'purple', 
                'red', 
                'green', 
                'brown', 
                'pink', 
                'yellow', 
                'gray', 
                'orange']

VALID_COHESIVENESS = ["True", "False"]