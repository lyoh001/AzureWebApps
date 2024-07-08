import calendar
import datetime
import io
import json
import os
import shutil
import tempfile
import warnings
from io import BytesIO
from pickle import load

import cv2
import numpy as np
import pandas as pd
import scipy.stats as stat
import tflite_runtime.interpreter as tflite
import uvicorn
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

warnings.filterwarnings("ignore")
load_dotenv(find_dotenv())
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
origins = [
    "https://lyoh001.com",
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


class QuestionRequest(BaseModel):
    question: str


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
        selected_month = [v for _, v in user_input.items()][0][0].split("-")[-1]
        if not selected_month:
            raise HTTPException(status_code=400, detail="Month is required")
        user_input = str(
            datetime.datetime.strptime(
                f"2023-{selected_month}-{calendar.monthrange(2023, int(selected_month))[-1]}",
                "%Y-%m-%d",
            )
        ).split()[0]
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
        return f"The AI model predicts that future Azure consumption for {selected_month}/2023 will be AUD ${df_future_pred.loc[user_input]['Bill']:,.2f}. This prediction is based on the data analysis and machine learning techniques that were used to optimize the license scheme and resource utilization. The model takes into account a number of factors, including the current usage patterns, the forecasted growth of the business, and the cost of Azure services. It is important to note that this is just a prediction and the actual cost may vary. However, the model has been shown to be accurate in the past and it is a valuable tool for businesses that are looking to reduce their Azure costs."

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Value Error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mlcoffeeplantdiseases", response_class=HTMLResponse)
async def get_mlcoffeeplantdiseases(request: Request):
    return templates.TemplateResponse(
        "mlcoffeeplantdiseases.html", {"request": request}
    )


@app.post("/mlcoffeeplantdiseases", response_class=HTMLResponse)
async def post_mlcoffeeplantdiseases(user_input: dict):
    try:
        model = tflite.Interpreter(
            model_path="models/mlcoffeeplantdiseases_model.tflite"
        )
        image_id = int(next(iter(user_input["image_id"])))
        actual = (
            "healthy"
            if image_id % 4 == 1
            else (
                "rust"
                if image_id % 4 == 2
                else "miner" if image_id % 4 == 3 else "phoma"
            )
        )
        preprocessed_image = (
            cv2.imread(f"static/data/mlcoffeeplantdiseases/{image_id}.jpg").reshape(
                1, 224, 224, -1
            )
            / 255.0
        )
        model.allocate_tensors()
        input_index = model.get_input_details()[0]["index"]
        input_tensor = preprocessed_image.astype("float32")
        output_details = model.get_output_details()
        model.set_tensor(input_index, input_tensor)
        model.invoke()
        prediction = ["healthy", "rust", "miner", "phoma"][
            np.argmax(model.get_tensor(output_details[0]["index"])[0])
        ]
        return f"The AI predicts that the coffee plant in the photo ({actual}) is {'not infected with any disease' if prediction == 'healthy' else f'infected with coffee leaf {prediction}, a fungal disease that can cause significant damage to coffee crops'}. The application was developed using a convolutional neural network (CNN) trained on a dataset of coffee plant images with and without diseases. The CNN can identify specific features of coffee plants affected by diseases, enabling it to predict infections. The application, currently under development, has shown promising results, achieving a 95% accuracy in correctly identifying diseased coffee plants. It can be a valuable tool for coffee farmers to identify and treat diseases in their crops and aid coffee researchers in studying disease effects on plants."

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Value Error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mlcovid", response_class=HTMLResponse)
async def get_mlcovid(request: Request):
    return templates.TemplateResponse("mlcovid.html", {"request": request})


@app.post("/mlcovid", response_class=HTMLResponse)
async def post_mlcovid(user_input: dict):
    try:
        model = tflite.Interpreter(model_path="models/mlcovid_model.tflite")
        image_id = int(next(iter(user_input["image_id"])))
        normal = [1, 2, 4, 7, 8, 10, 13, 16, 19, 20, 22, 25, 28]
        covid = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
        pneumonia = [5, 11, 14, 17, 23, 26, 29]
        actual = (
            "Normal"
            if image_id in normal
            else "Covid" if image_id in covid else "Pneumonia"
        )
        preprocessed_image = (
            cv2.imread(f"static/data/mlcovid/{image_id}.jpg").reshape(1, 224, 224, -1)
            / 255.0
        )
        model.allocate_tensors()
        input_index = model.get_input_details()[0]["index"]
        input_tensor = preprocessed_image.astype("float32")
        output_details = model.get_output_details()
        model.set_tensor(input_index, input_tensor)
        model.invoke()
        prediction = ["Normal", "Covid", "Pneumonia"][
            np.argmax(model.get_tensor(output_details[0]["index"])[0])
        ]
        return f"The AI predicts that the patient in the x-ray image ({actual}) is {'not infected with any disease' if prediction == 'Normal' else f'infected with {prediction}'}. Please note that the accuracy of the model will depend on the quality of the x-ray image and the size of the training dataset. If you are concerned about the accuracy of the prediction, you should consult with a medical professional."

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Value Error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mlmaternalmortality", response_class=HTMLResponse)
async def get_mlmaternalmortality(request: Request):
    return templates.TemplateResponse("mlmaternalmortality.html", {"request": request})


@app.post("/mlmaternalmortality", response_class=HTMLResponse)
async def post_mlmaternalmortality(user_input: dict):
    try:
        preprocessor = load(
            open("preprocessors/mlmaternalmortality_preprocessor.pkl", "rb")
        )
        model = tflite.Interpreter(model_path="models/mlmaternalmortality_model.tflite")
        payload = pd.DataFrame(
            {k: [np.nan] if next(iter(v)) == "" else v for k, v in user_input.items()},
            dtype="object",
        )
        model.allocate_tensors()
        input_index = model.get_input_details()[0]["index"]
        input_tensor = preprocessor.transform(payload).astype("float32")
        output_details = model.get_output_details()
        model.set_tensor(input_index, input_tensor)
        model.invoke()

        prediction = np.argmax(model.get_tensor(output_details[0]["index"]))
        age = user_input["Age"][0]
        blood_sugar = user_input["BS"][0]
        body_temp = user_input["BodyTemp"][0]
        diastolic_bp = user_input["DiastolicBP"][0]
        heart_rate = user_input["HeartRate"][0]
        systolic_bp = user_input["SystolicBP"][0]
        risk_level = "Low" if prediction == 0 else "Mid" if prediction == 1 else "High"
        return f"If the patient is {age} years old, has a blood sugar level of {blood_sugar} mg/dL, a body temperature of {body_temp} degrees Fahrenheit, a diastolic blood pressure of {diastolic_bp} mmHg, a heart rate of {heart_rate} beats per minute, and a systolic blood pressure of {systolic_bp} mmHg, the model would predict that the patient is at {risk_level} Risk for maternal complications. The model is still under development, but it has been shown to be effective in identifying potential health risks during pregnancy. The model can be used to help doctors and other healthcare providers make informed decisions about the care of pregnant women."

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Value Error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mlsupplychain", response_class=HTMLResponse)
async def get_mlsupplychain(request: Request):
    return templates.TemplateResponse("mlsupplychain.html", {"request": request})


@app.post("/mlsupplychain", response_class=HTMLResponse)
async def post_mlsupplychain(user_input: dict):
    try:
        df = pd.read_csv(
            "static/data/mlsupplychain/fact_order_lines_merged.csv",
            delimiter=",",
            thousands=",",
        )
        preprocessor_ifd = load(
            open("preprocessors/mlsupplychain_preprocessor_ifd.pkl", "rb")
        )
        preprocessor_otd = load(
            open("preprocessors/mlsupplychain_preprocessor_otd.pkl", "rb")
        )
        model_ifd = tflite.Interpreter(
            model_path="models/mlsupplychain_model_ifd.tflite"
        )
        model_otd = tflite.Interpreter(
            model_path="models/mlsupplychain_model_otd.tflite"
        )
        payload = {
            k: [np.nan] if next(iter(v)) == "" else v for k, v in user_input.items()
        }
        customer_id = int(next(iter(payload["customer_id"])))
        customer_name = df[df["customer_id"] == customer_id]["customer_name"].iloc[0]
        city = df[df["customer_id"] == customer_id]["city"].iloc[0]
        product_name = " ".join(
            s.capitalize() if i else s.upper()
            for i, s in enumerate(payload["product_name"][0].split())
        )
        category = df[df["product_name"] == product_name]["category"].iloc[0]
        sub_category = df[df["product_name"] == product_name]["sub_category"].iloc[0]
        order_qty = int(next(iter(payload["order_qty"])))
        infull_target = df[df["customer_id"] == customer_id]["infull_target%"].iloc[0]
        ontime_target = df[df["customer_id"] == customer_id]["ontime_target%"].iloc[0]
        order_date = datetime.datetime.strptime(payload["order_date"][0], "%Y-%m-%d")
        week_no = order_date.isocalendar()[1]
        days_for_delivery = int(next(iter(payload["days_for_delivery"])))
        total_order_qty = order_qty
        month = order_date.month
        day = order_date.day
        dayofweek = order_date.weekday()
        order_rate = 1

        X_ifd = pd.DataFrame(
            {
                "customer_id": [str(customer_id)],
                "customer_name": [customer_name],
                "city": [city],
                "product_name": [product_name],
                "category": [category],
                "sub_category": [sub_category],
                "order_qty": [order_qty**0.5],
                "infull_target%": [infull_target],
                "week_no": [week_no],
                "days_for_delivery": [days_for_delivery],
                "total_order_qty": [total_order_qty],
                "month": [month],
                "day": [day],
                "dayofweek": [dayofweek],
                "order_rate": [order_rate],
            }
        )
        X_otd = pd.DataFrame(
            {
                "customer_id": [str(customer_id)],
                "customer_name": [customer_name],
                "city": [city],
                "product_name": [product_name],
                "category": [category],
                "sub_category": [sub_category],
                "order_qty": [order_qty**0.5],
                "ontime_target%": [ontime_target],
                "week_no": [week_no],
                "days_for_delivery": [days_for_delivery],
                "total_order_qty": [total_order_qty],
                "month": [month],
                "day": [day],
                "dayofweek": [dayofweek],
                "order_rate": [order_rate],
            }
        )
        model_ifd.allocate_tensors()
        input_index = model_ifd.get_input_details()[0]["index"]
        input_tensor = preprocessor_ifd.transform(X_ifd).astype("float32")
        output_details = model_ifd.get_output_details()
        model_ifd.set_tensor(input_index, input_tensor)
        model_ifd.invoke()
        prediction_ifd = model_ifd.get_tensor(output_details[0]["index"])[0][0]
        model_otd.allocate_tensors()
        input_index = model_otd.get_input_details()[0]["index"]
        input_tensor = preprocessor_otd.transform(X_otd).astype("float32")
        output_details = model_otd.get_output_details()
        model_otd.set_tensor(input_index, input_tensor)
        model_otd.invoke()
        prediction_otd = model_otd.get_tensor(output_details[0]["index"])[0][0]
        return f"The AI model predicts that the probability of in-full delivery is {prediction_ifd * 100:.2f}% and the probability of on-time delivery is {prediction_otd * 100:.2f}%. These probabilities provide insights into the likelihood of successful and on-time delivery for the supply."

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Value Error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mltelecom", response_class=HTMLResponse)
async def get_mltelecom(request: Request):
    return templates.TemplateResponse("mltelecom.html", {"request": request})


@app.post("/mltelecom", response_class=HTMLResponse)
async def post_mltelecom(user_input: dict):
    try:
        preprocessor = load(open("preprocessors/mltelecom_preprocessor.pkl", "rb"))
        model = tflite.Interpreter(model_path="models/mltelecom_model.tflite")
        payload = {
            k: [np.nan] if next(iter(v)) == "" else v for k, v in user_input.items()
        }
        city_name = next(iter(payload["city_name"]))
        time_period = (
            datetime.datetime.strptime(payload["time_period"][0], "%Y-%m")
            - datetime.datetime.strptime("2022-09", "%Y-%m")
        ).days // 29
        data_size = float(next(iter(payload["data_size"])))
        valid_period = float(next(iter(payload["valid_period"])))
        X = pd.DataFrame(
            {
                "city_name": [city_name],
                "time_period": [time_period],
                "data_size": [data_size],
                "valid_period": [valid_period],
                "before/after_5g": ["After 5G"],
            }
        )
        X["data_size"] = stat.boxcox(X["data_size"].iloc[:1], -0.15236159025676418)[0]
        X["valid_period"] = stat.boxcox(X["valid_period"].iloc[:1], 0.2613605015797948)[
            0
        ]
        model.allocate_tensors()
        input_index = model.get_input_details()[0]["index"]
        input_tensor = preprocessor.transform(X).astype("float32")
        output_details = model.get_output_details()
        model.set_tensor(input_index, input_tensor)
        model.invoke()
        prediction = model.get_tensor(output_details[0]["index"])[0][0]
        return f"Based on the input attributes of city_name (ie '{city_name}'), data_size (ie '{data_size} Gb'), and valid_period (ie '{int(valid_period)} days'), the AI model predicts the monthly plan revenue to be ₹{prediction:,.2f} (Crores) post the 5G rollout. The model is trained on a dataset of historical data, and it uses this data to learn the relationships between the input attributes and the output. The accuracy of the model will depend on the quality of the training data. If the training data is representative of the real world, then the model will be more accurate. However, if the training data is not representative of the real world, then the model may be less accurate. The model can be used to help telecommunication companies make better decisions about their pricing and marketing strategies. For example, the model can be used to identify cities where there is a high demand for 5G plans, or to identify time periods when there is a seasonal increase in demand for mobile data."

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Value Error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mlvmaudit", response_class=HTMLResponse)
async def get_mlvmaudit(request: Request):
    return templates.TemplateResponse("mlvmaudit.html", {"request": request})


@app.get("/ollama", response_class=HTMLResponse)
async def get_ollama(request: Request):
    return templates.TemplateResponse("ollama.html", {"request": request})


@app.post("/ollama")
async def post_ollama(request: QuestionRequest):
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o",
        model="gpt-4o",
        temperature=0,
        streaming=True,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human", "{question}"),
        ]
    )
    output_parser = StrOutputParser()
    llm_chain = prompt | llm | output_parser
    stream = await llm_chain.ainvoke({"question": request.question})
    return StreamingResponse(stream, media_type="application/octet-stream")


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
                    "Write a concise overall comment for a student report card based on the following information. ```{question}```  Follow these three instructions.\n 1. Use Australian English spelling.\n 2. Do not use new lines.\n 3. Strictly never exceed 1200 characters in total.",
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


@app.post("/mlwklsgenerate")
async def post_mlwklsgenerate(user_input: dict):
    max_width = 492
    input_pdf_path = "input_pdf.pdf"
    output_pdf_path = "output_pdf_with_text.pdf"
    korean_font_path = "static/font/font.ttf"
    pdfmetrics.registerFont(TTFont("gulim", korean_font_path))
    try:
        student_name = user_input["studentName"]
        grade = int(user_input["grade"])
        attendance = user_input["attendance"]
        behaviour = user_input["behaviour"]
        effort = user_input["effort"]
        communication_skills = "\n".join(
            f"\u2022 {skill.split(']')[1]}"
            for skill in user_input["communicationSkills"]
        )
        understanding_skills = "\n".join(
            f"\u2022 {skill.split(']')[1]}"
            for skill in user_input["understandingSkills"]
        )
        overall_comment = user_input["overallComment"]
        src_path = "static/pdf"
        dst_path = os.path.join("/tmp", student_name)
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        for item in os.listdir(src_path):
            item_path = os.path.join(src_path, item)
            if os.path.isdir(item_path):
                shutil.copytree(item_path, os.path.join(dst_path, item))
            else:
                shutil.copy2(item_path, dst_path)

        pdf_reader = PdfReader(os.path.join(dst_path, input_pdf_path))
        pdf_writer = PdfWriter()

        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            packet = BytesIO()
            c = canvas.Canvas(packet)
            c.setFont("gulim", 11)

            c.drawString(155, 704, student_name)

            grade_text = f"{'Year' if grade else ''} {grade if grade else 'Prep'}"
            c.drawString(390, 704, grade_text)

            c.setFontSize(9)
            text_y = 652
            indent = 57
            bullet_indent = 67

            def draw_wrapped_text(text, start_y):
                for line in text.split("\n"):
                    words = line.split()
                    current_line = ""
                    first_line = True
                    for word in words:
                        test_line = current_line + word + " "
                        width = c.stringWidth(test_line, "gulim", 9)
                        if width <= max_width:
                            current_line = test_line
                        else:
                            c.drawString(
                                indent if first_line else bullet_indent,
                                start_y,
                                current_line.strip(),
                            )
                            start_y -= 11
                            current_line = word + " "
                            first_line = False
                    if current_line:
                        c.drawString(
                            indent if first_line else bullet_indent,
                            start_y,
                            current_line.strip(),
                        )
                        start_y -= 11
                return start_y

            text_y = draw_wrapped_text(communication_skills, text_y)
            text_y = 480
            text_y = draw_wrapped_text(understanding_skills, text_y)

            text_y = 305
            words = overall_comment.split()
            line = ""
            for word in words:
                test_line = line + word + " "
                width = c.stringWidth(test_line, "gulim", 9)
                if width <= max_width:
                    line = test_line
                else:
                    c.drawString(57, text_y, line.strip())
                    text_y -= 11
                    line = word + " "
            if line:
                c.drawString(57, text_y, line.strip())

            c.setFontSize(10)
            for score, y in zip([attendance, behaviour, effort], [119, 102, 84]):
                if score == "0":
                    text_x = 167
                elif score == "1":
                    text_x = 253
                elif score == "2":
                    text_x = 337
                elif score == "3":
                    text_x = 423
                else:
                    text_x = 509
                c.drawString(text_x, y, "X")

            c.save()
            packet.seek(0)
            modified_page = PdfReader(packet).pages[0]
            page.merge_page(modified_page)
            pdf_writer.add_page(page)

        output_path = os.path.join(dst_path, output_pdf_path)
        with open(output_path, "wb") as output_file:
            pdf_writer.write(output_file)

        if os.path.exists(output_path):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(open(output_path, "rb").read())

            return FileResponse(
                tmp_file.name, media_type="application/pdf", filename="report.pdf"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to generate PDF")

    except ValueError as e:
        return HTTPException(
            status_code=400,
            detail=f"Invalid input, please provide a valid user_input: {e}",
        )
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Error: {e}")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
