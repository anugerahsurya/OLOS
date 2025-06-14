# One Library for One Solution

Halo teman-teman, sudah lama tidak berjumpa. Gimana codingannya? seru pasti yaa. 

## Problem Definition

### What?
Nah kali ini aku bakal ngangkat masalah yang sering banget ditemuin karena cukup *annoying* banget nih. Yap, masalahnya berkaitan dengan pemanggilan kode yang berulang kali. Hal ini sering banget ditemuin nih, khususnya saat mengolah data tabular, karena pada dasarnya terdapat step-step yang wajib dilakukan sebelum masuk ke part paling diidolakan yaitu **Pemodelan**. Eitss, untuk itu mari kita masuk lebih dalam untuk tau kenapa itu bisa menjadi masalah.

### Why?
Pada dasarnya analisis yang mengikuti kerangka kerja mulai dari eksplorasi, encoding variabel, visualisasi, akan wajib dilakukan pada pengolahan data. Hal ini dikarenakan pemodelan pasti membutuhkan data yang sudah *clean* sehingga bisa diproses oleh model. Namun, hal ini seringkali dilakukan berulang dengan menyalin kode pada setiap cell saat proses debugging untuk menemukan perlakuan terbaik. Nah, bayangin nih kalo temen-temen bikin debugging banyak kemungkinan, berapa banyak line kode yang akan terbentuk. 

## Solution
Menurutku, kode yang bagus adalah kode yang ringkas, sehingga mudah dipahami tapi bisa ngasih insight yang banyak. Kalo di pemrograman kita mengenal istilah fungsi nih temen-temen. Kalo di matematika, fungsi itu digunakan untuk memetakan suatu variabel misal X ke suatu nilai Y. Nah, di program juga sama nih temen-temen. Fungsi dapat menerima input berupa parameter dan output yang merupakan keluaran dari fungsi tersebut. Fungsi tersebut bisa saja menerima 10 parameter dan mengeluarkan 1 output, ataupun sebaliknya. Hal ini akan berkaitan dengan variabel yang digunakan serta dihasilkan saat fungsi tersebut dijalankan.

***Jadi kenapa fungsi??*** Nah, dengan fungsi temen-temen bisa cukup mendefinisikan sekali suatu kode yang berkemungkinan dipanggil ulang saat pembuatan program, sehingga saat pemanggilan berulang nantinya temen-temen cukup memanggil fungsi tanpa harus menuliskan ulang semua kodenya. Pastinya membantu banget ga siii...


Berkaitan dengan fungsi, kali ini aku bakal ngespill sebuah kode yang bakal mempermudah pekerjaan teman-teman dalam masalah perkodingan. Tanpa berlama-lama disini aku bakal share beberapa fungsi yang ada di file **OLOS.py**. Fungsi ini dapat temen gunakan dengan mengunduh atau clone repository ini.

1. visualisasiData
Fungsi ini merupakan fungsi yang digunakan untuk membuat visualisasi data berupa Histogram dan Box-Plot untuk data bertipe numerik, serta Bar Chart untuk data bertipe Non Numerik. Fungsi ini menerima input berupa 1 parameter yaitu DataFrame. Fungsi ini tidak mengembalikan variabel, jadi outputnya berupa visualisasi langsung. Temen-temen dapat memahami contoh berikut.

```python
import pandas as pd
from OLOS import visualisasiData

# Load dan bersihkan data
data = pd.read_csv("Data/used_car_price_dataset_extended.csv").dropna()

# Visualisasi
visualisasiData(data)
```

2. korelasiVIFTarget
Fungsi ini merupakan fungsi yang digunakan untuk menghitung korelasi serta nilai ***Variance Inflation Factor (VIF)*** untuk setiap variabel. Perhitungan tersebut umum dijadikan dasar dalam feature selection karena mengacu pada keterkaitan antara variabel prediktor terhadap variabel target serta untuk pengecekan multikolinearitas. Fungsi ini menerima input berupa DataFrame dan nama variabel target. Fungsi ini mengembalikan 1 variabel yaitu DataFrame yang berisi daftar variabel beserta korelasinya terhadap variabel target dan nilai VIF-nya. Oiyaa, karena sementara korelasi digunakan adalah korelasi pearson, jadi fungsi ini hanya mengembalikan nilai untuk variabel bertipe numerik yaa.


```python
import pandas as pd
from OLOS import korelasiVIFTarget

# Load dan bersihkan data
df = pd.read_csv("Data/used_car_price_dataset_extended.csv").dropna()

target = 'usd_price'
korelasiVIFTarget(df, target)
```

3. Fungsi ini digunakan untuk melakukan encoding pada data kategorikal yang temen-temen punya. Seperti yang kita tahu, data kategorikal nggak bisa langsung dipakai di model machine learning karena model hanya bisa memahami angka. Nah, fungsi ini menyediakan tiga pilihan encoding, yaitu Label Encoding, One Hot Encoding, dan Ordinal Encoding, tergantung kebutuhan temen-temen. Kalau temen-temen pakai jenis=1, itu artinya temen-temen mau pakai Label Encoding, yaitu setiap kategori akan diubah jadi angka secara otomatis. Cocok nih buat target klasifikasi atau fitur yang kategorinya nggak punya urutan khusus. Kalau jenis=2, yang dilakukan adalah One Hot Encoding, yaitu setiap kategori akan diubah jadi kolom-kolom biner (0 atau 1). Ini biasanya dipakai kalau fitur kategorikalnya bersifat nominal, misalnya warna, jenis kelamin, dan lain-lain.Kalau jenis=3, itu buat temen-temen yang ingin pakai Ordinal Encoding, yaitu kategori diubah jadi angka berdasarkan urutan tertentu yang temen-temen tentukan sendiri lewat parameter ordinal_order. Cocok banget buat data kayak jenjang pendidikan atau level kepuasan. Fungsi ini juga secara otomatis nyimpen encoder-nya ke folder (default-nya "Encoder Tersimpan") supaya bisa dipakai lagi nanti, misalnya buat data uji atau proses deployment. Jadi, semua proses encoding-nya udah rapi dan bisa direplikasi dengan mudah.

```python
import pandas as pd
from OLOS import preprocessingData

# Load dan bersihkan data
df = pd.read_csv("Data/used_car_price_dataset_extended.csv").dropna()

# ===================== Label Encoder =========================
df, encoder_fuel = preprocessingData(data, 'transmission', jenis=1, save_dir='Encoder')
df, encoder_sh = preprocessingData(df, 'service_history', jenis=1, save_dir='Encoder')
df, encoder_ar = preprocessingData(df, 'accidents_reported', jenis=1, save_dir='Encoder')
df, encoder_iv = preprocessingData(df, 'insurance_valid', jenis=1, save_dir='Encoder')

# ===================== One Hot Encoder ============================
df, encoder_brand = preprocessingData(df, 'brand', jenis=2, save_dir='Encoder')
df, encoder_color = preprocessingData(df, 'color', jenis=2, save_dir='Encoder')


# Hanya contoh
urutan_fuel = ['Diesel','Petrol','Electric']
# ===================== Ordinal Encoder ==========================
df, encoder_fuel = preprocessingData(df, 'fuel_type', jenis=3, ordinal_order=urutan_fuel, save_dir='Encoder')

```

4. EliminasiOutlier
Fungsi ini merupakan fungsi yang digunakan untuk mengeliminasi outlier pada data yang temen-temen gunakan. Hal ini sangat berguna dalam proses pemodelan karena outlier dapat mempengaruhi kemampuan model dalam memprediksi data. Dengan menghilangkan outlier, temen-temen dapat meminimumkan error pada model yang dibangun karena kadang outlier merepresentasikan anomali pada data, sehingga kondisi tersebut bukan kondisi umum yang terjadi. Fungsi ini menggunakan algoritma **Isolation Forest** dalam proses eliminasi outlier karena kemampuannya yang bagus dalam deteksi anomali pada data [Jurnal Isolation Forest](http://dx.doi.org/10.1145/3338840.3355641). Fungsi ini menerima input berupa X_train dan y_train karena berkaitan dengan data yang akan diidentifikasi dan dilakukan eliminasi outlier, contamination yang merupakan persentase outlier dari data yang akan dieliminasi berkaitan dengan *hyperparameter* pada algoritma **Isolation Forest**, serta parameter case yang bernilai 1 untuk kasus regresi serta 2 untuk kasus klasifikasi. Pemrosesan untuk regresi dan klasifikasi dibedakan karena pada klasifikasi tentu karakteristik setiap class berbeda, sehingga pemrosesan harus dilakukan per class agar nilai suatu class tidak dianggap outlier jika diproses bersamaan.

```python
import pandas as pd
from OLOS import EliminasiOutlier

# Load dan bersihkan data
df = pd.read_csv("Data/used_car_price_dataset_extended.csv").dropna()

target = 'price_usd'
y = data[target]
X = df.drop(columns=target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# case 1: deteksi global
X_train, y_train = Eliminasi_Outlier(X_train.copy(), y_train.copy(), contamination=0.05, case=1)
```

5. eksplorasiBestFitur

Fungsi ini digunakan untuk mencari kombinasi fitur terbaik yang bisa temen-temen pakai dalam pemodelan, baik itu untuk regresi maupun klasifikasi. Jadi, temen-temen bisa menentukan beberapa fitur utama yang selalu dipakai, lalu fungsi ini akan secara otomatis menggabungkan fitur-fitur tambahan lainnya dalam berbagai kombinasi. Setiap kombinasi itu akan diuji performanya menggunakan beberapa model sekaligus seperti KNN, CatBoost, Random Forest, SVM, dan LightGBM. Fungsinya fleksibel banget karena bisa dipakai untuk dua jenis kasus. Kalau kasusnya regresi, evaluasinya pakai metrik RMSE. Tapi kalau klasifikasi, evaluasinya pakai F1-score macro. Jadi temen-temen tinggal tentuin aja case=1 untuk regresi atau case=2 untuk klasifikasi. Untuk setiap kombinasi fitur yang dicoba, fungsi ini akan membagi data jadi training dan testing, lalu fit beberapa model yang tersedia, dan menghitung skor metriknya. Dari situ dipilih model terbaik beserta nilai skornya. Semua hasilnya direkap dan disusun dalam bentuk tabel, lalu diurutkan dari yang performanya paling bagus. Dengan fungsi ini, temen-temen nggak perlu lagi capek-capek coba fitur satu-satu secara manual karena semuanya udah diotomatisasi, jadi proses eksplorasi fitur jadi jauh lebih efisien dan informatif.

```python
import pandas as pd
from OLOS import eksplorasiBestFitur

# Load dan bersihkan data
df = pd.read_csv("Data/used_car_price_dataset_extended.csv").dropna()

# ini merupakan daftar variabel yang ingin pasti digunakan dalam pemodelan, semakin sedikit ini maka semakin lama runningnya karena semakin banyak variabel yang dieksplorasi
fixed_features = ['make_year','mileage_kmpl','engine_cc','owner_count','transmission','service_history','accidents_reported','insurance_valid']

eksplorasiBestFitur(fixed_features, target, df, case=1)

```

## Reference

1. Cheng, Z., Zou, C., & Dong, J. (2019, September). Outlier detection using isolation forest and local outlier factor. In Proceedings of the conference on research in adaptive and convergent systems (pp. 161-168).
