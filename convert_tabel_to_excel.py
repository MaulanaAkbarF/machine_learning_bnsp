import json
import openpyxl

# Parse JSON string
json_string = '{"dataAlamatLokasiPengawasan":", Cibabat, Kecamatan Cimahi Utara, Kota Cimahi","dataComplianceLTHEFotoKemasanPath":null,"dataComplianceLTHEFotoSHEPath":null,"dataComplianceLTHEQuestion1Text":"Tidak","dataComplianceLTHEQuestion1aText":null,"dataComplianceLTHEQuestion2Text":"LTHE kurang jelas atau rusak karena tindakan produsen atau importir","dataComplianceLTHEQuestion3Text":"LTHE palsu","dataComplianceLTHEQuestion4":"","dataComplianceLTHEQuestion5Text":"LTHE tampaknya menunjukkan informasi yang benar tetapi memiliki masalah dengan visibilitas (disarankan tindak lanjuti dengan pengecer atau pemasok untuk tindakan perbaikan)","dataComplianceLTHEQuestion6":"","dataComplianceLainnyaQuestion1Text":"Ya","dataComplianceLainnyaQuestion2Text":"Ya","dataComplianceLainnyaQuestion3Text":"Ya","dataComplianceLainnyaQuestion4Text":"Ya","dataDetailProdukQuestion1":"118.001.4.01.104.16.0001","dataDetailProdukQuestion10Text":"4 Bintang - ⭐⭐⭐⭐","dataDetailProdukQuestion11":"3270.4","dataDetailProdukQuestion12":"4797676.80","dataDetailProdukQuestion13":"Rp. 999.999.999","dataDetailProdukQuestion14":"none","dataDetailProdukQuestion15":"PT Asus Indonesia","dataDetailProdukQuestion16":"Indonesia","dataDetailProdukQuestion2":"118.001.4.01.104.16.0001 - Samsung","dataDetailProdukQuestion3":"14/10/2022","dataDetailProdukQuestion4":"14/10/2025","dataDetailProdukQuestion5":"14/10/2025","dataDetailProdukQuestion6":"Samsung","dataDetailProdukQuestion7":"Indoor AR13KVFSD WKN / Outdoor AR13KVFSD WKX","dataDetailProdukQuestion8":"Indoor AR13KVFSD WKN / Outdoor AR13KVFSD WKX","dataDetailProdukQuestion9Text":"Inverter","dataDetailProdukQuestionDetail1":"1120","dataDetailProdukQuestionDetail2":"12000","dataDetailProdukQuestionDetail3":"11.7","dataDetailProdukQuestionDetail4Text":"CSPF","dataDetailProdukQuestionDetail5":"R32","dataEnablePickupSampleRecordingQuestion1Text":null,"dataEnablePickupSampleRecordingQuestion2Text":null,"dataEnablePickupSampleRecordingQuestion3Text":null,"dataEnablePickupSampleRecordingQuestion4Text":null,"dataEnablePickupSampleRecordingQuestion5Text":null,"dataFotoDataRitelPath":null,"dataLatitude":"-6.8822166","dataLongitude":"107.5561701","dataNamaLokasiPengawasan":"tes","dataNamaTenagaPenjual":"tes2","dataPengalamanLTHEQuestion1Text":"Familiar","dataPengalamanLTHEQuestion2Text":"Sering","dataPengalamanLTHEQuestion3Text":"Simpan produk sampai produk pengganti berlabel diterima dari pemasok","dataPengalamanLTHEQuestion4Text":"Pernah","dataPengalamanLTHEQuestion5Text":"Sebagian","dataStatus1":"success","dataStatus2":"","dataStatus3":"","dataStatusPengawasan":"found","dataTanggalPengawasan":"02/10/24 / 20:26:18","dataWilayahLokasiPengawasan":"Jawa Barat","id":1}'
data = json.loads(json_string)

# Ambil nama atribut (kunci)
attributes = list(data.keys())

# Buat workbook Excel baru
wb = openpyxl.Workbook()
ws = wb.active

# Tulis nama atribut ke Excel, satu per baris
for i, attr in enumerate(attributes, start=1):
    ws.cell(row=i, column=1, value=attr)

# Simpan file Excel
wb.save('attribute_names10.xlsx')

print("Nama atribut telah disimpan ke dalam file 'attribute_names.xlsx'")