import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import datetime
import sys

# 사용자 정의 simglucose 사용
sys.path.insert(0, './simglucose')

# 데이터 불러오기
df = pd.read_csv("adolescent#001_100_500.csv")
df["Time"] = pd.to_datetime(df["Time"])

# 그래프 시각화
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Time"], y=df["BG"], name="실제 혈당", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=df["Time"], y=df["CGM"], name="CGM", line=dict(color="green")))
fig.add_trace(go.Scatter(x=df["Time"], y=df["insulin"], name="인슐린 주입량", yaxis="y2", line=dict(color="red")))

fig.update_layout(
    title="혈당 & 인슐린 추이",
    xaxis_title="시간",
    yaxis=dict(title="혈당 (mg/dL)"),
    yaxis2=dict(title="인슐린 (U)", overlaying="y", side="right")
)
st.plotly_chart(fig, use_container_width=True)

# 특정 시점 데이터 추출
reference_index = 250
context_window = 10
context_df = df.iloc[reference_index - context_window: reference_index + 1][["Time", "BG", "CGM", "CHO", "insulin"]]
ai_insulin = df.iloc[reference_index + 1]["insulin"]
ai_response_df = df.iloc[reference_index + 1: reference_index + 31][["Time", "BG", "CGM"]]

st.subheader("AI 시뮬레이션 기반 정보")
st.write("최근 상황 요약", context_df.tail(1))
st.write(f"💉 AI 추천 주입량: {ai_insulin:.2f}U")
st.write("AI 주입 후 예상 혈당 변화", ai_response_df.head())

# 사용자 시뮬레이션 실행
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.controller.base import Action

sensor = CGMSensor.withName("Dexcom")
pump = InsulinPump.withName("Insulet")
patient = T1DPatient.withName("adolescent#001")
scenario = RandomScenario(start_time=datetime.datetime.now(), seed=42)
env = T1DSimEnv(patient=patient, sensor=sensor, pump=pump, scenario=scenario)
state = env.reset()

custom_basal = 0.5
insulin_doses = [0.5, 0.4, 0.3, 0.1, 0.3]

bg_list, cgm_list, insulin_list = [], [], []

for dose in insulin_doses:
    obs, reward, done, info = env.step(Action(basal=custom_basal, bolus=dose))
    bg_list.append(obs[0])
    cgm_list.append(obs[0])
    insulin_list.append(dose)

sim_result_df = pd.DataFrame({
    "Time Step": list(range(len(insulin_doses))),
    "BG": bg_list,
    "CGM": cgm_list,
    "Insulin": insulin_list
})

st.subheader("🧪 사용자 시뮬레이션 결과")
st.line_chart(sim_result_df.set_index("Time Step"))
