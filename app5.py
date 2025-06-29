import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import datetime
import sys
import plotly.graph_objects as go
import copy
from plotly.subplots import make_subplots

# 사용자 정의 simglucose 경로 설정
sys.path.insert(0, './simglucose')

from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.controller.base import Action

components.html("""
    <script>
        window.scrollTo({ top: 0, behavior: 'smooth' });
    </script>
""", height=0)

import random

def get_random_persona(group: str, weight: float):
    # 페르소나 사전
    persona_dict = {
        "청소년": [
            {"id":"p1", "name": "민석", "gender": "남", "weight_range": (55, 75), "desc": "운동을 좋아하는 고등학생", "emoji": "👦"},
            {"id":"p2","name": "하린", "gender": "여", "weight_range": (35, 55), "desc": "소식가, 과일 위주 식단", "emoji": "👧"},
            {"id":"p3","name": "지후", "gender": "남", "weight_range": (40, 60), "desc": "아침 자주 거르고 부모가 관리", "emoji": "👦"},
        ],

        "성인": [
            {"id":"p4","name": "재훈", "gender": "남", "weight_range": (70, 110), "desc": "앉아서 일하는 직장인", "emoji": "👨"},
            {"id":"p5","name": "지민", "gender": "여", "weight_range": (60, 85), "desc": "주부, 간식 자주 먹음", "emoji": "👩"},
            {"id":"p6","name": "보미", "gender": "여", "weight_range": (45, 65), "desc": "운동 강사, 고강도 운동", "emoji": "👩"},
        ]
    }

    if group not in persona_dict:
        return None

    # 첫 번째 조건을 만족하는 페르소나 하나 반환
    for p in persona_dict[group]:
        if p["weight_range"][0] <= weight <= p["weight_range"][1]:
            return p

    return None  # 해당 조건 만족하는 페르소나 없음

def plot_static(fig, **kwargs):
    st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True, **kwargs})
    
def analyze_glucose_events(bg_series, time_series):
    df_g = pd.DataFrame({"time": time_series, "bg": bg_series})
    df_g["status"] = "정상"
    df_g.loc[df_g["bg"] < 70, "status"] = "저혈당"
    df_g.loc[df_g["bg"] > 180, "status"] = "고혈당"

    df_g["group"] = (df_g["status"] != df_g["status"].shift()).cumsum()
    events = df_g[df_g["status"] != "정상"].groupby("group")

    messages = []
    for _, group in events:
        status = group["status"].iloc[0]
        t_start = group["time"].iloc[0].strftime("%H:%M")
        t_end = group["time"].iloc[-1].strftime("%H:%M")
        messages.append(f"- **{t_start} ~ {t_end}** 사이에 **{status}** 발생")

    return messages, df_g

if st.session_state.get("trigger_scroll", False):
    components.html("""
        <script>
            window.scrollTo({ top: 0, behavior: 'smooth' });
        </script>
    """, height=0)
    st.session_state.trigger_scroll = False  # 플래그 초기화
    
# 세션 상태 초기화
if "step" not in st.session_state:
    st.session_state.step = 0
if "selected_patient" not in st.session_state:
    st.session_state.selected_patient = None

st.title("🩺 인슐린 제어 시뮬레이터")

# STEP 0: 환자 선택
if st.session_state.step == 0:

    st.image("how.png", use_container_width=True)

    st.markdown("""
        <div style='text-align: center; font-size: 36px; font-weight: bold; padding: 20px 0;'>
        당신은 오늘,<br> 당뇨병 환자가 되어 하루를 살아갑니다
        </div>

        <div style='text-align: center; font-size: 22px; color: gray;'>
        하루 동안 어떤 선택이 혈당을 어떻게 바꿀까요?<br>
        직접 혈당을 확인하고, 식사와 인슐린을 조절해 보세요.
        </div>
        """, unsafe_allow_html=True)

    st.subheader("환자 선택")
    patient_name = st.selectbox("알아보고 싶은 환자를 선택하세요:", [
        "adult#001", "adult#002", "adult#003","adult#004", "adult#005",
        "adolescent#001", "adolescent#002", "adolescent#003","adolescent#004", "adolescent#005",
    ])
    
    if st.button("다음 단계로"):
        st.session_state.selected_patient = patient_name
        st.session_state.csv_file = f"{patient_name}_100_500.csv"
        st.session_state.step = 1
        st.rerun()

# STEP 1: 데이터 불러오기
elif st.session_state.step == 1:
    df = pd.read_csv(f"data/{st.session_state.csv_file}")
    df["Time"] = pd.to_datetime(df["Time"])
    st.session_state.session_df = df
    st.write(f"✅ {st.session_state.selected_patient} 데이터 로딩 완료")

    # CSV 불러오기 (앱 시작 시 한 번만 실행되게 outside에 둘 수도 있음)
    df_params = pd.read_csv("vpatient_params.csv")  # 경로에 맞게 수정

    # 선택된 환자 이름
    patient_name = st.session_state.selected_patient
    
    # 해당 환자 데이터 필터링
    patient_info = df_params[df_params["Name"] == patient_name]


        # 정보가 존재할 경우 출력
    if not patient_info.empty:
        info = patient_info.iloc[0]

        # 그룹 분류
        if "adolescent" in patient_name:
            group = "청소년"
        elif "adult" in patient_name:
            group = "성인"
        else:
            group = "기타"
    else:
        st.warning("선택한 환자의 정보를 찾을 수 없습니다.")

    # 몸무게와 그룹 기반 페르소나 선택
    selected_persona = get_random_persona(group, info["BW"])
    image_path = f"patient_images/{selected_persona['id']}.png"

    persona_story_dict = {
        "p1": "“코트 위에서 땀 흘릴 땐 아무 생각 없어요.\n하지만 운동 끝나고 쓰러질 듯한 저혈당이 찾아올 땐,\n왜 나만 이런 조심이 필요한 걸까 싶어요.\n그래도... 좋아하는 농구를 계속하려면,\n인슐린과 친구가 되어야겠죠.”",
        "p2": "“아침엔 배가 고프지 않아요.\n그냥 달콤한 귤 한 쪽이면 충분하죠.\n하지만 나도 몰랐어요. 과일이 그렇게 혈당을 올릴 줄은...\n이젠 작은 한 입도 조심스럽지만,\n건강을 위해 선택해야 할 일이 생겼어요.”",
        "p3": "“내 혈당 수첩은 늘 엄마 손에 있어요.\n게임보다 더 어려운 게 인슐린 타이밍 맞추기예요.\n아침을 거르고 학교에 가면…\n갑자기 어지러워지는 건 아직도 무서워요.\n조금씩, 혼자서도 할 수 있겠죠?”",
        "p4": "“점심은 대충 컵라면,\n회의 끝나고 커피에 달달한 쿠키 하나.\n그 뒤로 느껴지는 몸의 무거움과 피곤함…\n일이 바빠 혈당을 챙기지 못한 날엔,\n내 몸이 말없이 경고를 보내요.\n이제는 신호를 그냥 넘기지 않기로 했어요.”",
        "p5": "“아이가 밥을 남기면 그걸로 제 점심이 돼요.\n빵 한 조각, 과자 몇 개로 하루를 넘기곤 하죠.\n하지만 어느 날, 이유 없는 어지럼증이 찾아왔어요.\n아이를 지키기 위해선,\n먼저 내 건강을 지켜야 한다는 걸 알았어요.”",
        "p6": "“하루 종일 에너지 넘치는 나,\n하지만 수업 직전엔 속이 울렁이고 손이 떨려요.\n운동은 내 삶의 일부지만,\n그만큼 철저한 관리도 함께 따라와요.\n건강한 몸과 마음을 위해,\n오늘도 내 몸의 신호에 귀를 기울여요.”",
    }


    if selected_persona:
        persona_id = selected_persona["id"]
        story = persona_story_dict.get(persona_id, "")
        image_path = f"patient_images/{persona_id}.png"  # 사전에 생성한 Sora 이미지

        st.markdown(f"""
            <div style='font-size: 28px; font-weight: bold; margin-bottom: 10px;'>
            👤 선택된 환자: {selected_persona['emoji']} {selected_persona['name']}
            </div>

            <div style='font-size: 20px; line-height: 1.6;'>
            <ul style='list-style: none; padding-left: 0;'>
            <li><strong>성별:</strong> {selected_persona['gender']}</li>
            <li><strong>설명:</strong> {selected_persona['desc']}</li>
            <li><strong>몸무게 (BW):</strong> {info['BW']:.1f} kg</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

            
        st.image(image_path)
        st.markdown(f"""
        <div style='padding: 20px; background-color: #f8f9fa; border-radius: 12px; border: 1px solid #ddd; box-shadow: 2px 2px 8px rgba(0,0,0,0.05);'>
            <p style='font-size: 18px; font-weight: bold; color: #444; margin-bottom: 12px;'>📝 환자 이야기</p>
            <p style='font-size: 16px; color: #333; line-height: 1.8; white-space: pre-line;'>
                {story}
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("해당 몸무게에 적절한 페르소나가 없습니다.")
    st.markdown("")
    if st.button("➡️ 시뮬레이션 시작"):
        st.session_state.step = 21
        st.session_state.persona_id = persona_id  # 여기서 명시적으로 저장
        st.rerun()

for seg in [1, 2, 3]:
    df = st.session_state.get("session_df")
    if df is None:
        # st.error("❌ 데이터가 아직 로딩되지 않았습니다. 처음부터 다시 시도해 주세요.")
        st.stop()
    section_df = df.iloc[seg * 160 : (seg + 1) * 160].reset_index(drop=True)
    bg_now = section_df["BG"].iloc[0]
    meal_total = section_df["CHO"].sum()
    persona_id = st.session_state.get("persona_id", "p1") 

    dose_key = f"dose{seg}"
    basal_key = f"basal{seg}"
    bg_key = f"bg_user{seg}"
    env_key = "env_user"
    env_result_key = f"env_result_{seg}"
    env_init_key = f"env_state_{seg}"

    # 🧷 키가 없을 경우 기본값으로 초기화
    st.session_state.setdefault(dose_key, 1.0)
    st.session_state.setdefault(basal_key, 0.02)

    bg_step = 20 + (seg - 1) * 4 + 1
    meal_step = bg_step + 1
    input_step = bg_step + 2
    result_step = bg_step + 3

    # 예: seg는 1~3 중 하나
    section_index = seg - 1  # 또는 st.session_state.step에서 계산
    STEP_PER_SECTION = 160
    start = section_index * STEP_PER_SECTION
    end = start + STEP_PER_SECTION
    section_start_dt = pd.to_datetime(df["Time"].iloc[start])
    section_end_dt = pd.to_datetime(df["Time"].iloc[end - 1])
    start_time_str = section_start_dt.strftime("%H:%M")
    end_time_str = section_end_dt.strftime("%H:%M")


    # 혈당 확인 UI 출력
    if st.session_state.step == bg_step:
        st.image("BG.png")
        # 혈당 상태 분류
        if bg_now < 70:
            status_label = "저혈당"
            status_message = "⚠️ 혈당이 낮아요. 몸이 떨리거나 어지럽지 않으신가요?"
            status_color = "#0288d1"
        elif bg_now > 180:
            status_label = "고혈당"
            status_message = "⚠️ 혈당이 높아요. 갈증이나 피로감이 느껴질 수 있어요."
            status_color = "#e64a19"
        else:
            status_label = "정상"
            status_message = "✅ 혈당이 안정적이에요. 지금 상태를 유지해 볼까요?"
            status_color = "#388e3c"

        st.markdown(f"### ⏱️ {start_time_str} ~ {end_time_str} 구간")
        st.markdown(f"""
                <div style='padding: 14px 16px; background-color: #f5f5f5; border-radius: 10px; border-left: 6px solid {status_color};'>
                    <p style='font-size: 22px; font-weight: 600; color: {status_color}; margin-bottom: 8px;'>
                        현재 혈당 상태: <b style='color:{status_color};'>{bg_now:.1f} mg/dL</b> {status_label} 
                    </p>
                    <p style='font-size: 18px; color: #333;'>{status_message}</p>
                    ※ 정상 혈당 범위는 <b>70~180 mg/dL</b>입니다.
                </div>
            """, unsafe_allow_html=True)
        st.markdown("")

        if st.button("➡️ 식사 확인"):
            st.session_state.step += 1
            st.rerun()

    # 2단계: 식사 확인
    elif st.session_state.step == meal_step:
        st.image("meal.png")
        st.subheader(f"📈 {seg}/3구간 - 식사 확인")
        st.markdown(f"⏱️시간대: **{start_time_str} ~ {end_time_str}**")    
        # st.subheader(f"🍽️ {seg}구간 - 식사 확인")
        # st.markdown(f"이번 구간 섭취량: **{meal_total:.1f} g**")
        meal_df = df.iloc[start:end][df["CHO"] > 0]

        if not meal_df.empty:
            # st.markdown("### 🍽 식사 정보") 
            for _, row in meal_df.iterrows():
                time = pd.to_datetime(row["Time"]).strftime("%H:%M")
                cho = round(row["CHO"], 1)

                # 음식 예시와 칼로리 추정인
                if cho < 10:
                    food = "딸기 한 줌 🍓 / 우유 1컵 🥛"
                elif cho < 20:
                    food = "바나나 1개 🍌 / 고구마 반 개 🍠"
                elif cho < 30:
                    food = "식빵 1.5장 🍞 / 그래놀라 요거트 🥣"
                elif cho < 40:
                    food = "공기밥 반 공기 🍚 / 토스트 세트 🍳"
                elif cho < 55:
                    food = "라면 1개 🍜 / 김밥 1줄 🍙"
                elif cho < 70:
                    food = "한식 도시락 🍱 / 떡볶이 + 순대 🍢"
                else:
                    food = "햄버거 세트 🍔🍟 / 피자 2조각 🍕"

                kcal = int(cho * 4)
                # 카드 형식 출력 (가독성 강조)
                st.markdown(f"""
                <div style='padding: 16px; margin-bottom: 14px; background-color: #fffaf0;
                            border-left: 6px solid #FFA94D; border-radius: 10px;'>
                    <p style='font-size: 18px; color: #333; margin-bottom: 6px;'>
                        <b>🕒 {time} 식사</b>
                    </p>
                    <p style='font-size: 16px; color: #444; margin: 0;'>
                        {food}<br>
                        <span style='color: #FF6F00; font-weight: 600;'>탄수화물 {cho}g → 약 {kcal} kcal</span>
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='padding: 16px; background-color: #f9f9f9; border-radius: 10px;'>
                <p style='font-size: 16px; color: #555;'>🥛 <b>공복 상태:</b> 해당 시간에는 식사 기록이 없어요.</p>
            </div>
            """, unsafe_allow_html=True)

        if st.button("➡️ 인슐린 입력"):
            st.session_state.step += 1
            st.rerun()

    # 3단계: 인슐린 입력
    elif st.session_state.step == input_step:
        st.subheader(f"💉 {seg}구간/3 - 인슐린 조절")
        st.markdown(f"⏱️시간대: **{start_time_str} ~ {end_time_str}**")    
       

        target_bg = 110
        gf = 50
        icr = 10
        meal_carb = df.iloc[start:end]["CHO"].sum()
        correction = max((bg_now - target_bg), 0) / gf
        meal_insulin = meal_carb / icr
        recommended_bolus = round(correction + meal_insulin, 2)

        # 인슐린 권장 정보 카드 출력
        st.markdown(f"""
        <div style='padding: 18px; background-color: #eef8f6; border-radius: 12px;
                    border-left: 6px solid #20c997; margin-bottom: 16px;'>
            <p style='font-size: 18px; margin-bottom: 8px; color: #333;'>
                💉 <b>현재 상태에 따른 권장 인슐린 주입 정보</b>
            </p>
            <ul style='font-size: 16px; line-height: 1.8; color: #444; padding-left: 20px;'>
                <li><b>현재 혈당:</b> {bg_now:.1f} mg/dL</li>
                <li><b>감도 계수 (GF):</b> {gf}</li>
                <li><b>탄수화물 당 인슐린 비율 (ICR):</b> {icr}</li>
                <li><b>예상 식사 탄수화물:</b> {meal_carb:.1f} g</li>
                <li><b>권장 볼루스 인슐린(식사시 주입):</b> <span style='color:#e67700; font-weight:600;'>{recommended_bolus} 단위</span></li>
                <li><b>권장 기저 인슐린(평소에 주입) (8시간 기준):</b> <span style='color:#1c7ed6;'>0.03 ~ 0.04 단위/3분</span></li>
            </ul>
            <p style='font-size: 14px; color: #777; margin-top: 8px;'>
                ⚠️ 이 수치는 시뮬레이션을 위한 권장 값으로, 실제 인슐린 처방은 의료 전문가의 지시를 따라야 합니다.
            </p>
        </div>
        """, unsafe_allow_html=True)


        st.markdown("권장량을 참고해 슬라이더를 움직여 인슐린 주입량을 조절해 보세요")
        dose = st.slider("볼루스 인슐린 조절 (식사시 주입)", 0.0, 5.0, value=st.session_state[dose_key], key=dose_key)
        basal = st.slider("기저 인슐린 조절 (평소에 주입)", 0.0, 0.05, value=st.session_state[basal_key], step=0.001, key=basal_key)

        if seg == 1 and env_key not in st.session_state:
            sensor = CGMSensor.withName("Dexcom")
            pump = InsulinPump.withName("Insulet")
            patient = T1DPatient.withName(st.session_state.selected_patient, init_state=bg_now)
            scenario = RandomScenario(start_time=datetime.datetime.now(), seed=42)
            env_user = T1DSimEnv(patient=patient, sensor=sensor, pump=pump, scenario=scenario)
            env_user.reset()
            st.session_state[env_key] = env_user

        if env_init_key not in st.session_state:
            st.session_state[env_init_key] = copy.deepcopy(st.session_state[env_key])

        if st.button("📊 시뮬레이션 실행"):
            st.session_state.step += 1
            st.rerun()

    # 4단계: 결과 분석
    elif st.session_state.step == result_step:
        st.subheader(f"📈 {seg}/3 구간 - 결과 분석")
        st.markdown(f"⏱️시간대: **{start_time_str} ~ {end_time_str}**")    
       

        dose = st.session_state.get(dose_key, 1.0)
        basal = st.session_state.get(basal_key, 0.02)
        env = copy.deepcopy(st.session_state[env_init_key])
        result = []

        meal_times = section_df[section_df["CHO"] >= 10].index.tolist()
        bolus_step = max(meal_times[0] - 10, 0) if meal_times else None

        for t in range(160):
            bolus = dose if bolus_step == t else 0.0
            obs, _, _, _ = env.step(Action(basal=basal, bolus=bolus))
            result.append(obs[0])

        st.session_state[bg_key] = result
        st.session_state[env_result_key] = copy.deepcopy(env)

        # 시각화
        # ⏱ x축: 시작 시간 + 3분 간격 × 스텝
        start_time = pd.to_datetime(df["Time"].iloc[start])
        time_range = [start_time + datetime.timedelta(minutes=3 * i) for i in range(160)]

        # time_range = [datetime.datetime.strptime("00:00", "%H:%M") + datetime.timedelta(minutes=(seg * 160 + i) * 3) for i in range(160)]
        fig = go.Figure()
        fig.add_shape(type="rect", xref="x", yref="y", x0=time_range[0], x1=time_range[-1], y0=70, y1=180,
                      fillcolor="green", opacity=0.2, layer="below", line_width=0)
        fig.add_trace(go.Scatter(x=time_range, y=result, mode="lines", name="혈당", line=dict(color="red"), yaxis="y1"))
        fig.add_trace(go.Bar(x=time_range, y=section_df["CHO"], name="식사량", marker_color="lightblue", yaxis="y2"))
        fig.update_layout(
            title=f"구간 {seg}/3 혈당 및 식사량",
            xaxis=dict(title="시간", tickangle=45),
            yaxis=dict(title="혈당", side="left", range=[40, max(result) + 20]),
            yaxis2=dict(title="식사량", overlaying="y", side="right"),
            legend=dict(x=0.01, y=1.1, orientation="h")
        )
        plot_static(fig)

        # 시뮬레이션 결과 시각화 후 이어서 실행
        from PIL import Image
        import os

        # 분석 함수 사용
        messages, df_g = analyze_glucose_events(result, time_range)

        # 혈당 이벤트 메시지 표시
        if messages:
            st.markdown("#### ⛑️ 혈당 이벤트 감지")
            for msg in messages:
                st.markdown(msg)
        else:
            st.success("모든 구간에서 정상 혈당을 유지했어요!")

        # 마지막 상태에 따라 이미지 선택
        last_status = df_g["status"].iloc[-1]

        # 고혈당/저혈당 발생 여부에 따라 이미지 선택
        statuses = df_g["status"].unique()
        img_suffix = ""

        if "고혈당" in statuses:
            img_suffix = "-1"
        elif "저혈당" in statuses:
            img_suffix = "-2"


        img_path = f"./patient_images/{persona_id}{img_suffix}.png"
        # 이미지 출력
        if os.path.exists(img_path):
            st.image(Image.open(img_path), caption=f"현재 상태: {last_status}", use_container_width=True)
        else:
            st.warning("해당 상태에 맞는 이미지가 없습니다.")

        bg_final = result[-1]

        if st.button("➡️ 다음 구간으로"):
            st.session_state.env_user = copy.deepcopy(st.session_state[env_result_key])
            st.session_state.dose_basal = basal
            st.session_state.step += 1
            st.rerun()

if st.session_state.step == 33:
    st.header("📊 하루 요약 리포트")
    tir_values = []
    for seg in [1, 2, 3]:
        bg = st.session_state.get(f"bg_user{seg}", [])
        if bg:
            tir = sum(70 <= g <= 180 for g in bg) / len(bg) * 100
            tir_values.append(tir)
            st.markdown(f"- {seg}구간 TIR: **{tir:.1f}%**")
    if tir_values:
        avg = sum(tir_values) / len(tir_values)
        st.success(f"👉 전체 평균 TIR: **{avg:.1f}%**")
