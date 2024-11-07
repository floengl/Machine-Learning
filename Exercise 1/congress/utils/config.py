import os


class Config:
    BASE_DIR = "V:/OneDrive/Dokumente/Uni/ML/Exercise 1/Exercise 1/congress"
    DATA_DIR = os.path.join(BASE_DIR, "data")
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    SUBMISSION_DIR = os.path.join(BASE_DIR, "submissions")
    PLOTS_DIR = os.path.join(BASE_DIR, "plots")
