# คู่มือการใช้งาน Template สำหรับ VRP Solver

## ภาพรวม

ไฟล์ `vrp_input_template.xlsx` เป็น Template สำหรับกรอกข้อมูลนำเข้าให้กับระบบ VRP Solver ประกอบด้วย 5 Sheet ดังนี้:

| Sheet | คำอธิบาย |
|-------|----------|
| Instructions | คำแนะนำการใช้งาน |
| Nodes | ข้อมูลจุดเก็บขยะ |
| Vehicles | ข้อมูลรถเก็บขยะ |
| Distance_Matrix | ระยะทางระหว่างจุด |
| Settings | การตั้งค่าจุดเริ่มต้น |

---

## วิธีการกรอกข้อมูล

### 1. Sheet: Nodes (ข้อมูลจุดเก็บขยะ)

กรอกข้อมูลจุดเก็บขยะทั้งหมดในพื้นที่

| คอลัมน์ | คำอธิบาย | ตัวอย่าง |
|---------|----------|----------|
| Node ID | รหัสจุด (เลขจำนวนเต็ม เริ่มจาก 1) | 1, 2, 3, ... |
| Node Name | ชื่อจุด/สถานที่ | "จุดเริ่มต้น", "หมู่บ้าน A" |
| General Demand | ปริมาณขยะทั่วไป (หน่วย) | 50, 80, 0 |
| Recycle Demand | ปริมาณขยะรีไซเคิล (หน่วย) | 5, 10, 0 |
| Node Type | ประเภทจุด | depot, customer, checkpoint |

**ประเภทจุด (Node Type):**
- `depot` - จุดเริ่มต้น (ต้องเป็น Node ID = 1)
- `customer` - จุดเก็บขยะทั่วไป
- `checkpoint` - จุดทิ้งขยะ (ต้องมีอย่างน้อย 1 จุด)

**ข้อกำหนดสำคัญ:**
- Node ID = 1 **ต้องเป็น depot เสมอ**
- depot และ checkpoint ควรมี demand = 0
- ต้องมี checkpoint อย่างน้อย 1 จุด

---

### 2. Sheet: Vehicles (ข้อมูลรถเก็บขยะ)

กรอกข้อมูลประเภทรถที่ใช้งาน

| คอลัมน์ | คำอธิบาย | ตัวอย่าง |
|---------|----------|----------|
| Vehicle Type | รหัสประเภทรถ | A, B, C |
| General Capacity | ความจุขยะทั่วไป (หน่วย) | 2000 |
| Recycle Capacity | ความจุขยะรีไซเคิล (หน่วย) | 200 |
| Fixed Cost (THB) | ค่าใช้จ่ายคงที่ต่อคัน (บาท) | 2400 |
| Fuel Cost (THB/km) | ค่าน้ำมันต่อกิโลเมตร (บาท/กม.) | 8, 8.5, 9 |

---

### 3. Sheet: Distance_Matrix (ระยะทางระหว่างจุด)

กรอกระยะทางระหว่างจุดทั้งหมดในรูปแบบ Matrix

**รูปแบบ:**
```
From/To |  1   |  2   |  3   |  4   |  5
--------|------|------|------|------|------
   1    |  0   | 1500 | 2300 | 1800 | 3000
   2    | 1500 |  0   | 1200 | 2100 | 2500
   3    | 2300 | 1200 |  0   |  900 | 1800
   4    | 1800 | 2100 |  900 |  0   | 2200
   5    | 3000 | 2500 | 1800 | 2200 |  0
```

**ข้อกำหนดสำคัญ:**
- หน่วยเป็น **เมตร** (ไม่ใช่กิโลเมตร)
- Matrix ต้องเป็น **Symmetric** (ระยะจาก i→j = j→i)
- แนวทแยง (Diagonal) ต้องเป็น 0
- ต้องครบทุกจุด (ถ้ามี 10 จุด ต้องเป็น Matrix 10x10)

---

### 4. Sheet: Settings (การตั้งค่า)

| Setting | Value | คำอธิบาย |
|---------|-------|----------|
| Depot Node ID | 1 | รหัสจุดเริ่มต้น (ปกติคือ 1) |
| Depot Name | Depot | ชื่อจุดเริ่มต้น |

---

## ขั้นตอนการใช้งาน

### ขั้นตอนที่ 1: กรอกข้อมูล

1. เปิดไฟล์ `vrp_input_template.xlsx`
2. **ลบข้อมูลตัวอย่าง** (แถวสีเขียว) ใน Sheet: Nodes, Vehicles, Distance_Matrix
3. กรอกข้อมูลจริงของคุณ
4. บันทึกไฟล์ (เช่น `my_data.xlsx`)

### ขั้นตอนที่ 2: แปลงเป็น JSON

เปิด Command Prompt หรือ Terminal แล้วรันคำสั่ง:

```bash
python convert_template_to_json.py my_data.xlsx
```

หรือระบุชื่อไฟล์ output:

```bash
python convert_template_to_json.py my_data.xlsx my_output.json
```

### ขั้นตอนที่ 3: รัน VRP Solver

ใช้ไฟล์ JSON ที่ได้กับ VRP Solver:

```bash
python vrp_solver_v2.py my_output.json
```

---

## ตัวอย่างข้อมูล

### ตัวอย่าง Nodes (5 จุด)

| Node ID | Node Name | General Demand | Recycle Demand | Node Type |
|---------|-----------|----------------|----------------|-----------|
| 1 | จุดเริ่มต้น | 0 | 0 | depot |
| 2 | หมู่บ้าน A | 50 | 5 | customer |
| 3 | หมู่บ้าน B | 80 | 10 | customer |
| 4 | หมู่บ้าน C | 30 | 3 | customer |
| 5 | จุดทิ้งขยะ | 0 | 0 | checkpoint |

### ตัวอย่าง Vehicles (3 ประเภท)

| Vehicle Type | General Capacity | Recycle Capacity | Fixed Cost | Fuel Cost |
|--------------|------------------|------------------|------------|-----------|
| A | 2000 | 200 | 2400 | 8.0 |
| B | 2000 | 200 | 2400 | 8.5 |
| C | 2000 | 200 | 2400 | 9.0 |

---

## ข้อผิดพลาดที่พบบ่อย

| ข้อผิดพลาด | สาเหตุ | วิธีแก้ไข |
|------------|--------|-----------|
| "Depot node not found" | ไม่พบ Node ID ที่ตรงกับ Depot Node ID ใน Settings | ตรวจสอบว่า Node ID = 1 มีอยู่ใน Sheet Nodes |
| "No checkpoint found" | ไม่มีจุด checkpoint | เพิ่มจุดที่มี Node Type = "checkpoint" |
| "Matrix not symmetric" | ระยะทาง i→j ไม่เท่ากับ j→i | ตรวจสอบและแก้ไข Distance Matrix |
| "Matrix size mismatch" | จำนวนจุดใน Matrix ไม่ตรงกับ Nodes | ตรวจสอบให้ Matrix มีขนาดเท่ากับจำนวน Nodes |

---

## ไฟล์ที่เกี่ยวข้อง

| ไฟล์ | คำอธิบาย |
|------|----------|
| `vrp_input_template.xlsx` | Template สำหรับกรอกข้อมูล |
| `convert_template_to_json.py` | Script แปลง Excel เป็น JSON |
| `create_template_v2.py` | Script สร้าง Template ใหม่ |
| `vrp_solver_v2.py` | โปรแกรม VRP Solver หลัก |

---

## ติดต่อสอบถาม

หากพบปัญหาในการใช้งาน กรุณาตรวจสอบ:
1. รูปแบบข้อมูลถูกต้องตามที่กำหนด
2. ไม่มีเซลล์ว่างในข้อมูลที่จำเป็น
3. Distance Matrix เป็น Symmetric และมีขนาดถูกต้อง
