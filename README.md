# Resnet-Exploration_300Gecs

# plain34 loss and accuracy history graph
![plain graph](https://i.imgur.com/i9eIJQA.png )

# resNET34 loss and accuracy history graph
![resnet graph](https://i.imgur.com/2Cc2pZl.png)

Model Polos (Plain Model)

Akurasi: Garis akurasi pelatihan (biru) terus meningkat hingga mendekati 100%, namun garis akurasi validasi (oranye) stagnan dan bahkan sedikit menurun setelah epoch ke-20.
Loss: Garis loss pelatihan (biru) terus menurun, tetapi garis loss validasi (oranye) mulai naik setelah epoch ke-20.
Kesimpulan: Ini adalah tanda klasik dari overfitting overfitting. Artinya, model menjadi terlalu "hafal" dengan data pelatihan, tetapi tidak mampu menggeneralisasi dengan baik pada data baru (data validasi). Kinerjanya buruk di dunia nyata.

Model ResNet

Akurasi: Garis akurasi pelatihan dan validasi sama-sama naik secara konsisten dan nilainya sangat berdekatan. Keduanya mencapai akurasi yang lebih baik dari lain .
Loss: Garis loss pelatihan dan validasi sama-sama menurun dan tetap rendah.
Kesimpulan: Ini menunjukkan generalisasi yang bagus. Model tidak hanya belajar dari data pelatihan, tetapi juga mampu memberikan prediksi yang akurat pada data yang belum pernah dilihat sebelumnya. Ini adalah ciri model yang robust dan andal.

# plain34 confussion matrix
![confusion matrix plain](https://i.imgur.com/2qmzniu.png)

Kegagalan Total pada Soto Ayam: Model sepenuhnya "buta" terhadap kelas soto_ayam. Sebagian besar gambar soto_ayam salah diidentifikasi sebagai bakso (20 kali) dan gado_gado (13 kali). Ini menunjukkan kemungkinan adanya kemiripan visual yang sangat tinggi (misalnya, makanan berkuah dalam mangkuk) yang tidak dapat dibedakan oleh model.Kebingungan Antara Makanan Berkuah: Ada kebingungan besar antara bakso, gado_gado (yang terkadang disajikan dengan kuah kacang), dan soto_ayam. Model cenderung "memilih" bakso sebagai tebakan default untuk gambar-gambar ini.Kebingungan Rendang dan Nasi Goreng: Model cukup sering salah mengira rendang sebagai nasi_goreng (15 kali). Ini bisa jadi karena palet warna yang serupa (kecoklatan) atau cara penyajian dalam dataset.

# resnet34 confussion matrix
![confusion matrix resnet](https://i.imgur.com/1YHalql.png)

Peningkatan TerbesaradalahKelas soto_ayam yang sebelumnya gagal total, kini berhasil dikenali dengan cukup baik (29 dari 39 gambar benar). Ini adalah perbaikan paling signifikan.Kelas Terbaik menjadi Nasi Goreng (49 benar) dan Rendang (37 benar) menunjukkan kinerja yang solid dan menjadi kelas yang paling andal bagi model ini.Terdapat masalah baru yaitu Kinerja pada kelas bakso dan gado_gado justru menurun drastis. Keduanya sekarang sangat sering salah diklasifikasikan sebagai soto_ayam.



## Berikut adalah hyperparamter yang digunakan untuk komperasi antara 2 model (plain34 vs resnet34):
### Hyperparameters
    num_epochs = 10
    batch_size = 24
    learning_rate = 0.0005
    weight_decay = 0.0005

Pada saat mulai training, validation accuracy pada plain34 di epochs 1/10 22.86, kemudian turun menjadi 15.9% di epochs 2, sedangkan pada model resnet34 validation accuracy pada epochs 1 38.37%, dan mengalami penurunan hingga epochs 3 hingga menjadi 32.2%. Pada plain34 hanya mengalami kenaikan pada epochs 2 epochs saja, yaitu 3 dan 7 sedangkan pada resnet34 mengalami kenaikan pada epochs 4, 5, 6, dan 8. 

## plain34

![plain](https://i.imgur.com/jczcNoN.png)


Pada plain34 training memiliki validation accuracy sebesar 22.86% saat training dimulai, akurasi validasi terupdate 3 kali selama 10 epoch di mana akurasi validasi terbesar berada di 54,29%

## resnet34

![resnet](https://i.imgur.com/fcpe8QK.png)

Pada resnet34 training memiliki akurasi validasi 38,47% saat training dimulai, akurasi validasi terupdate 5 kali selama 10 epoch dimana akurasi validasi terbesar berada di 60%


Pada kedua gambar diatas dapat dilihat jika kenaikan nilai accuracy pada resnet34 lebih stabil jika dibandingkan dengan plain34, hal ini dikarenakan resnet terdapat skip connections. Skip connections memungkinkan gradien pada layer sebelumnya untuk melewati beberapa lapisan dan langsung ditambahkan ke lapisan yang lebih dalam. Skip connections tersebut yang membuat model resnet ini akurasinya terus meningkat selama training yang kita lakukan.

