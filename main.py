from backend.main import app


if __name__ == "__main__":
    import uvicorn

    root_dir = "/home/samuel/Documents/AiMlAuto"

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[f"{root_dir}/backend", f"{root_dir}/app", f"{root_dir}/frontend"],
        reload_excludes=[f"{root_dir}/uploads"],
    )