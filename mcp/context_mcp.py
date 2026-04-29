import subprocess
from datetime import datetime
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# Khởi tạo MCP Server
mcp = FastMCP("MedicalAI_Context_Server")


@mcp.tool()
def get_recent_git_history(num_commits: int = 5) -> str:
    """Lấy lịch sử các thay đổi (commit) gần nhất của dự án."""
    try:
        result = subprocess.run(
            ['git', 'log', f'-n {num_commits}', '--stat'],
            capture_output=True, text=True, check=True
        )
        return result.stdout
    except Exception as e:
        return f"Lỗi khi đọc Git history: {str(e)}"


@mcp.tool()
def get_current_changes() -> str:
    """Lấy các đoạn code vừa bị sửa nhưng chưa được lưu vào Git (diff)."""
    try:
        result = subprocess.run(
            ['git', 'diff'],
            capture_output=True, text=True, check=True
        )
        return result.stdout
    except Exception as e:
        return f"Lỗi khi đọc Git diff: {str(e)}"


@mcp.tool()
def generate_and_save_report() -> str:
    """
    Tổng hợp Git history và Git diff hiện tại, sau đó lưu thành file báo cáo .md
    trong thư mục 'regular_report' với tên file là ngày giờ hiện tại.
    """
    try:
        # Lấy dữ liệu từ 2 tool trên
        history = get_recent_git_history(5)
        diff = get_current_changes()

        # Tạo nội dung báo cáo
        report_content = f"# BÁO CÁO TIẾN ĐỘ DỰ ÁN MEDICAL AI\n"
        report_content += f"**Ngày tạo:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        report_content += "## 1. Các Commit Gần Đây\n```text\n"
        report_content += history + "\n```\n\n"

        report_content += "## 2. Các Thay Đổi Chưa Commit (Working Tree)\n```diff\n"
        report_content += diff if diff.strip() else "Không có thay đổi nào chưa commit."
        report_content += "\n```\n"

        # Đảm bảo thư mục regular_report tồn tại
        report_dir = Path("regular_report")
        report_dir.mkdir(parents=True, exist_ok=True)

        # Tạo tên file theo định dạng: YYYY-MM-DD_HH-MM-SS.md
        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".md"
        filepath = report_dir / filename

        # Lưu file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_content)

        return f"Thành công! Đã lưu báo cáo tại: {filepath.absolute()}"

    except Exception as e:
        return f"Lỗi trong quá trình tạo báo cáo: {str(e)}"


if __name__ == "__main__":
    # Chạy server qua giao thức stdio để Continue có thể giao tiếp được
    mcp.run(transport="stdio")