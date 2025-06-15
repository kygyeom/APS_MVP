# 🩸 Insulin Control Simulator

An interactive web-based simulator for understanding insulin therapy in Type 1 Diabetes (T1D) patients. Users can adjust basal and bolus insulin manually and compare the glucose control results with an AI-driven insulin strategy.

---

## 📌 Purpose

- Simulate real-life glucose control experiences of people with diabetes.
- Demonstrate the importance of **insulin timing** and **dose selection**.
- Help users understand the roles of **basal** and **bolus** insulin.
- Provide **visual and analytical feedback** on blood glucose levels.
- Allow comparison with an **AI-based controller** for optimal regulation.

---

## 🎯 Key Features

- ✅ Manual basal & bolus insulin dose control
- 🍽 Meal detection and pre-meal bolus suggestion (30 minutes before)
- 📈 Real-time glucose visualization with normal range (70–180 mg/dL)
- 🤖 AI insulin control comparison
- 🩸 Hypo-/hyperglycemia event analysis
- 📊 Summary report with final evaluation
- 📚 Educational expanders (What is basal/bolus insulin? What is TIR?)

---

## 🛠 Tech Stack

| Component   | Technology    |
|-------------|---------------|
| Frontend    | Streamlit     |
| Backend     | SimGlucose, Python |
| Visualization | Plotly     |
| Environment | Conda + YAML setup |

---

## 🔧 How to Run

```bash
# Clone the repository
git clone https://github.com/your-username/insulin-simulator.git
cd insulin-simulator

# Create environment
conda env create -f environment.yaml
conda activate insulin-sim

# Launch Streamlit app
streamlit run app.py

🧠 Simulation Flow
Blood Glucose Measurement

Meal Detection and CHO Estimation

Insulin Injection (Basal / Bolus)

Activity and Glucose Control Outcome

All steps are visualized to help users understand the impact of each decision.

💬 Example Feedback
"Final BG: 195 mg/dL. Consider increasing basal insulin or reducing meal CHO."

"Blood glucose spike detected after lunch. Pre-meal bolus timing may be delayed."

📚 Educational Use
This simulator is ideal for:

Healthcare learners and professionals

Diabetes educators

AI in medicine researchers

Public health demonstrations

📢 License
This project is for educational and non-commercial research purposes. Contact the maintainer for inquiries.

👨‍⚕️ Credits
Created by [Your Name] at Kongju National University
Part of the [AI-integrated Artificial Pancreas Project]

go
Copy code
