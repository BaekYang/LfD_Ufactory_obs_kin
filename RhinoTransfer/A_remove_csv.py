import os
import csv
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Remove rows from first y<200 point to end')
    parser.add_argument('--file_name', type=str, required=True,
                        help='input CSV filename (without .csv or with) located under data/')
    args = parser.parse_args()
    # 입력 파일 경로 설정
    fname = args.file_name
    if not fname.lower().endswith('.csv'):
        fname += '.csv'
    input_path = os.path.join('data', fname)
    if not os.path.isfile(input_path):
        print(f"[ERROR] 입력 파일을 찾을 수 없습니다: {input_path}")
        sys.exit(1)

    # 출력 파일명 생성 (원본명_remove.csv)
    base, ext = os.path.splitext(fname)
    output_name = f"{base}_remove{ext}"
    output_path = os.path.join('data', output_name)

    # CSV 읽기
    with open(input_path, 'r', encoding='utf-8', newline='') as fin:
        reader = csv.DictReader(fin)
        rows = list(reader)
        fieldnames = reader.fieldnames

    # y < 200인 첫 번째 인덱스 찾기
    cut_index = None
    for idx, row in enumerate(rows):
        try:
            y_val = float(row.get('y', 0))
        except ValueError:
            continue
        if y_val < 200:
            cut_index = idx
            break

    # 삭제 범위 설정: cut_index부터 끝까지 삭제
    if cut_index is not None:
        filtered = rows[:cut_index]
        print(f"[INFO] y<200인 첫 행 발견 (인덱스 {cut_index}), 이후 {len(rows)-cut_index}행 삭제됨.")
    else:
        filtered = rows
        print("[INFO] y<200인 행이 없어 모든 행을 유지합니다.")

    # 결과 CSV 쓰기
    with open(output_path, 'w', encoding='utf-8', newline='') as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered)
    print(f"[INFO] 필터링된 CSV가 저장되었습니다: {output_path}")

if __name__ == '__main__':
    main()
