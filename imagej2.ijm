PATH = "/beamlinedata/BMIT/projects/prj37G13104/rec/2023-7-26-ID/XFDing_cochlear/2142L_001/scan_000-4splits-rec/set1";
rangestart = 495;
rangeend = 505;

mkDir(PATH+"/A-test");
run("Image Sequence...", "open="+PATH+"/A/sli-000-0000.tif sort use");
run("Duplicate...", "duplicate range="+rangestart+"-"+rangeend);
run("Image Sequence... ", "format=TIFF use save="+PATH+"/A-test/0000.tif");
closeAll();

mkDir(PATH+"/B-test");
run("Image Sequence...", "open="+PATH+"/B/sli-000-0000.tif sort use");
run("Duplicate...", "duplicate range="+rangestart+"-"+rangeend);
run("Image Sequence... ", "format=TIFF use save="+PATH+"/B-test/0000.tif");

closeAll();
print("Finished");

function leftPad(n, width) {
        s =""+n;
        while (lengthOf(s)<width)
                s = "0"+s;
        return s;
}
function rightPad(n, width) {
        s =""+n;
        while (lengthOf(s)<width)
                s = s+"0";
        return s;
}
function decimalPad(f,widthA,widthB) {
        s = ""+f;
        pos = 0;
        for (loop=0; loop<=(lengthOf(s)-1); loop+=1) {
                if (fromCharCode(charCodeAt(s,loop)) == ".") {
                        pos = loop;
                }
        }
        if (pos == 0) {
                outputstr = ""+leftPad(f,widthA)+"-"+leftPad(0,widthB);
        } else {
                strA = "";
                strB = "";
                for (loopA=0; loopA<=(pos-1); loopA+=1) {
                        strA = strA+fromCharCode(charCodeAt(s,loopA));
                }
                for (loopB=(pos+1); loopB<=(lengthOf(s)-1); loopB+=1) {
                        strB = strB+fromCharCode(charCodeAt(s,loopB));
                }
                outputstr = ""+leftPad(strA,widthA)+"-"+rightPad(strB,widthB);
        }
        return outputstr;
}
function mkDir(f) {
        if (File.exists(f) == 1) {
                print("dir: "+f+" exits");
        } else {
                print("creating: "+f);
                File.makeDirectory(f);
        }
}
function closeAll() {
        while (nImages()>0) {
                selectImage(nImages());
                run("Close");
        }
}
