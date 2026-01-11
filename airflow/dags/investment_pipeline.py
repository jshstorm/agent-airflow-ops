from datetime import datetime, timedelta
import os
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

# 호스트 머신의 프로젝트 절대 경로 (Docker Volume Mount용)
# .env에서 AIRFLOW_HOST_PROJECT_PATH를 설정해야 합니다.
# 기본값은 로컬 개발 환경 기준입니다.
HOST_PROJECT_PATH = os.getenv("AIRFLOW_HOST_PROJECT_PATH", "/Users/jshstorm/Documents/github/agent")

default_args = {
    'owner': 'sisyphus',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'investment_daily_analysis',
    default_args=default_args,
    description='Investment Analysis Pipeline (US -> KR -> Dividend)',
    schedule_interval='0 22 * * 1-5',  # 평일 밤 10시
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['investment', 'stock', 'ops'],
) as dag:

    # 1. 미국 주식 분석 (US Stock Analysis)
    # update_all.py 실행
    us_analysis = DockerOperator(
        task_id='us_stock_analysis',
        image='investment-analyst:latest',
        api_version='auto',
        auto_remove=True,
        command="python implementations/USStockAnalysis/update_all.py",
        mounts=[
            # 소스코드 마운트 (Host Path -> Container Path)
            Mount(source=f"{HOST_PROJECT_PATH}", target="/app", type="bind"),
        ],
        # 데이터 저장을 위한 추가 볼륨이 있다면 여기에 추가
        environment={
            'DATA_DIR': '/app/implementations/USStockAnalysis/data'
        },
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        # 타임아웃 30분
        execution_timeout=timedelta(minutes=30),
    )

    # 2. 한국 주식 분석 (KR Stock Analysis)
    # 미국장 분석 완료 후 실행 (데이터 의존성은 없으나 리소스 분산 목적)
    kr_analysis = DockerOperator(
        task_id='kr_stock_analysis',
        image='investment-analyst:latest',
        api_version='auto',
        auto_remove=True,
        command="python implementations/StockAI/run_analysis.py",
        mounts=[
            Mount(source=f"{HOST_PROJECT_PATH}", target="/app", type="bind"),
        ],
        environment={
            'KR_DATA_DIR': '/app/implementations/StockAI/data'
        },
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        execution_timeout=timedelta(minutes=30),
    )

    # 3. 배당 포트폴리오 최적화 (Dividend Optimizer)
    # 독립적으로 실행 가능하나, 구조상 마지막에 배치
    dividend_opt = DockerOperator(
        task_id='dividend_optimization',
        image='investment-analyst:latest',
        api_version='auto',
        auto_remove=True,
        command="python implementations/DividendOptimizer/optimizer_job.py", # 가상의 진입점 (추후 구현 필요)
        mounts=[
            Mount(source=f"{HOST_PROJECT_PATH}", target="/app", type="bind"),
        ],
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
    )

    # 실행 순서 정의
    # US 분석 -> KR 분석 -> 배당 최적화
    us_analysis >> kr_analysis >> dividend_opt
