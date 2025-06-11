import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import sys

# 사용자 정의 simglucose 경로 설정
sys.path.insert(0, './simglucose')

from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.controller.base import Action

# 세션 상태 초기화
if "step" not in st.session_state:
    st.session_state.step = 0
if "selected_patient" not in st.session_state:
    st.session_state.selected_patient = None

st.title("🩺 인슐린 제어 시뮬레이터")

# STEP 0: 환자 선택
if st.session_state.step == 0:
    st.subheader("1️⃣ 환자 선택")
    patient_name = st.selectbox("시뮬레이션할 환자를 선택하세요:", [
        "adolescent#001", "adolescent#002",
        "adult#001"
    ])
    csv_file = f"{patient_name}_100_500.csv"

    if st.button("다음 단계로"):
        st.session_state.selected_patient = patient_name
        st.session_state.csv_file = csv_file
        st.session_state.step += 1
        st.rerun()

# STEP 1: 사용자 인슐린 제어 설정
elif st.session_state.step == 1:
    st.subheader("2️⃣ 사용자 인슐린 제어 설정 (6시간)")

    # 사용자 인슐린 용량 설정
    dose1 = st.slider("0~2시간", 0.0, 0.5, 0.1, 0.01)
    dose2 = st.slider("2~4시간", 0.0, 0.5, 0.1, 0.01)
    dose3 = st.slider("4~6시간", 0.0, 0.5, 0.1, 0.01)
    st.session_state.doses = [dose1]*40 + [dose2]*40 + [dose3]*40

    # 데이터 로드
    df = pd.read_csv(st.session_state.csv_file)
    df["Time"] = pd.to_datetime(df["Time"])

    # 초기 혈당 설정 (AI 기준)
    mid_index = len(df) // 2
    ai_df = df.iloc[mid_index - 60: mid_index + 60].reset_index(drop=True)
    init_bg = ai_df["BG"].values[0]
    # init_bg = 140 

    # 사용자 시뮬레이션 환경 구성
    sensor = CGMSensor.withName("Dexcom")
    pump = InsulinPump.withName("Insulet")
    base_patient = T1DPatient.withName(st.session_state.selected_patient)

    patient = T1DPatient.withName(
        st.session_state.selected_patient,
        init_state=init_bg
    )
    scenario = RandomScenario(start_time=datetime.datetime.now(), seed=42)
    env_user = T1DSimEnv(patient=patient, sensor=sensor, pump=pump, scenario=scenario)
    env_user.reset()

    # 사용자 시뮬레이션 실행
    bg_user, ins_user = [], []
    for u in st.session_state.doses:
        obs, _, _, _ = env_user.step(Action(basal=0.0, bolus=u))
        bg_user.append(obs[0])
        ins_user.append(u)

    # AI 데이터 추출
    bg_ai = ai_df["BG"].values
    ins_ai = ai_df["insulin"].values

    # 결과 시각화
    st.subheader("📈 혈당 비교: AI vs 사용자")
    fig_bg = go.Figure()
    fig_bg.add_trace(go.Scatter(y=bg_ai, name="AI 혈당", line=dict(color="green")))
    fig_bg.add_trace(go.Scatter(y=bg_user, name="사용자 혈당", line=dict(color="blue")))
    fig_bg.update_layout(
        xaxis_title="Time Step (3min)",
        yaxis_title="혈당 (mg/dL)",
        height=400
    )
    st.plotly_chart(fig_bg, use_container_width=True)

    st.subheader("💉 인슐린 주입량 비교")
    fig_insulin = go.Figure()
    fig_insulin.add_trace(go.Scatter(y=ins_ai, name="AI 인슐린", line=dict(color="orange", dash="dash")))
    fig_insulin.add_trace(go.Scatter(y=ins_user, name="사용자 인슐린", line=dict(color="red")))
    fig_insulin.update_layout(
        xaxis_title="Time Step (3min)",
        yaxis_title="인슐린 (U)",
        height=300
    )
    st.plotly_chart(fig_insulin, use_container_width=True)

    # 다음 단계로 이동할 버튼
    if st.button("➡️ 다음 단계로 (TIR 비교)"):
        st.session_state.bg_user = bg_user
        st.session_state.ins_user = ins_user
        st.session_state.bg_ai = bg_ai
        st.session_state.ins_ai = ins_ai
        st.session_state.step += 1
        st.rerun()

# STEP 2: 시뮬레이션 및 결과
elif st.session_state.step == 2:


    # 3. TIR 계산 및 막대 시각화
    def compute_tir(bg_series):
        in_range = np.logical_and(np.array(bg_series) >= 70, np.array(bg_series) <= 180)
        return 100 * np.sum(in_range) / len(bg_series)

    tir_ai = compute_tir(st.session_state.bg_ai)
    tir_user = compute_tir(st.session_state.bg_user)

    fig_tir = go.Figure()
    fig_tir.add_trace(go.Bar(
        x=["AI", "사용자"],
        y=[tir_ai, tir_user],
        marker_color=["green", "blue"]
    ))
    fig_tir.update_layout(
        title="TIR (Time in Range) 비교",
        yaxis_title="TIR (%)",
        xaxis_title="제어 주체",
        yaxis=dict(range=[0, 100]),
        height=400
    )
    st.plotly_chart(fig_tir, use_container_width=True)

    if st.button("처음으로 돌아가기"):
        st.session_state.step = 0
        st.rerun()
