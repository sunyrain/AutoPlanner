import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
names = z.namelist()
print("count:", len(names))
for n in names[:15]:
    print(" ", n)
print("total uncompressed:", sum(i.file_size for i in z.infolist()))
