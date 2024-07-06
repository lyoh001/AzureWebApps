import calendar
import datetime
import io
import json
import os
from pickle import load

import numpy as np
import pandas as pd
import tflite_runtime.interpreter as tflite
import uvicorn
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

load_dotenv(find_dotenv())
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
origins = [
    "https://domain.com",
    "https://azurewebapp.azurewebsites.net",
    "http://localhost:3000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/status")
async def status():
    return {"status": 200}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/mlcloudaudit", response_class=HTMLResponse)
async def get_mlcloudaudit(request: Request):
    return templates.TemplateResponse("mlcloudaudit.html", {"request": request})


@app.post("/mlcloudaudit", response_class=HTMLResponse)
async def post_mlcloudaudit(user_input: dict):
    try:
        X_preprocessor = load(
            open("preprocessors/mlcloudaudit_X_preprocessor.pkl", "rb")
        )
        y_preprocessor = load(
            open("preprocessors/mlcloudaudit_y_preprocessor.pkl", "rb")
        )
        model = tflite.Interpreter(model_path="models/mlcloudaudit_model.tflite")

        selected_month = user_input.get("month")
        if not selected_month:
            raise HTTPException(status_code=400, detail="Month is required")

        selected_month = selected_month.split("-")[-1]
        user_input_date = datetime.datetime.strptime(
            f"2023-{selected_month}-{calendar.monthrange(2023, int(selected_month))[-1]}",
            "%Y-%m-%d",
        ).date()

        N_PAST = 5
        N_FUTURE = 1
        ASFREQ = "MS"
        FREQ = "1m"
        DATE_COL = "Date"
        y_label = "Bill"

        df = pd.read_csv(
            io.StringIO(
                """Date,Bill
            2022-06,334230
            2022-07,344321
            2022-08,351373
            2022-09,346814
            2022-10,387434
            2022-11,394204
            2022-12,401308"""
            )
        )
        df[DATE_COL] = pd.to_datetime(df[DATE_COL].str.strip(), format="%Y-%m")
        df = df.set_index(DATE_COL).sort_index().asfreq(ASFREQ)

        X_test_scaled = X_preprocessor.transform(df)
        X_test = np.array(
            [
                X_test_scaled[i : i + N_PAST]
                for i in range(len(X_test_scaled) - N_PAST - N_FUTURE + 1)
            ]
        )

        pred_future, current_batch = [], X_test[-1:]
        for _ in range(N_PAST):
            model.allocate_tensors()
            input_index = model.get_input_details()[0]["index"]
            input_tensor = current_batch.astype("float32")
            output_details = model.get_output_details()
            model.set_tensor(input_index, input_tensor)
            model.invoke()
            current_pred = model.get_tensor(output_details[0]["index"])[0]
            pred_future.append(current_pred)
            current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

        df_future_pred = pd.DataFrame(
            y_preprocessor.inverse_transform(pred_future),
            columns=[y_label],
            index=pd.date_range(df.index[-1], periods=N_PAST, freq=FREQ, name=DATE_COL),
        )

        prediction = (
            f"The AI model predicts that future Azure consumption for {selected_month}/2023 "
            f"will be AUD ${df_future_pred.loc[user_input_date]['Bill']:,.2f}. This prediction is based on the data analysis "
            "and machine learning techniques that were used to optimize the license scheme and resource utilization. "
            "The model takes into account a number of factors, including the current usage patterns, the forecasted growth "
            "of the business, and the cost of Azure services. It is important to note that this is just a prediction and the "
            "actual cost may vary. However, the model has been shown to be accurate in the past and it is a valuable tool for "
            "businesses that are looking to reduce their Azure costs."
        )

        return prediction

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Value Error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/wkls", response_class=HTMLResponse)
async def get_wkls(request: Request):
    return templates.TemplateResponse("wkls.html", {"request": request})


@app.post("/mlwklsbrush", response_class=JSONResponse)
async def post_mlwklsbrush(user_input: dict):
    try:
        ratings = {
            "0": "Very Low",
            "1": "Needs Attention",
            "2": "Satisfactory",
            "3": "High",
            "4": "Excellent",
        }
        data = {
            "student_name": user_input["studentName"],
            "grade": f"{grade}" if (grade := int(user_input["grade"])) else "Prep",
            "attendance": ratings.get(user_input["attendance"], "Unknown"),
            "behaviour": ratings.get(user_input["behaviour"], "Unknown"),
            "effort": ratings.get(user_input["effort"], "Unknown"),
            "communication_skills": " and ".join(
                skill.split("]")[1] for skill in user_input["communicationSkills"]
            ),
            "understanding_skills": " and ".join(
                skill.split("]")[1] for skill in user_input["understandingSkills"]
            ),
        }

        llm = AzureChatOpenAI(
            azure_deployment=os.environ["OPENAI_DEPLOYMENT_NAME"],
            model=os.environ["OPENAI_MODEL_NAME"],
            temperature=float(os.environ["TEMPERATURE"]),
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a Korean teacher at a primary school in Australia, and your task is to complete the end-of-year student report card. Here is a sample report that you can refer to: ```Jenny has made great progress this semester! She is a kind and gentle member of our class. She has grown in self-confidence throughout the semester. Jenny contributes to class discussions, is attentive, and always follows instructions. She confidently seeks clarification if she does not understand what is expected. She has made connections and friendships with her peers. She can work in groups and independently. Overall, Jenny has made a great start to Grade 1! Keep it up, Jenny!```",
                ),
                (
                    "human",
                    "Write a concise overall comment for a student report card based on the following information. ```{question}```  Follow these three instructions.\n 1. Use Australian English spelling.\n 2. Do not use new lines.\n 3. Do not exceed 1500 characters in total.",
                ),
            ]
        )
        question = json.dumps(data)
        llm_chain = prompt | llm
        return {"updatedComment": llm_chain.invoke({"question": question}).content}

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing key: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mlwklsgenerate", response_class=JSONResponse)
async def post_mlwklsgenerate(user_input: dict):
    try:
        updated_comment = user_input
        return {"updatedComment": updated_comment}
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing key: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
