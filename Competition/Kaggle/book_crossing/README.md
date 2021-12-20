```
1、user: "User-ID";"Location";"Age"
User-ID作为一个维度处理
Location: 行如："stockton, california, usa"，分别是City, state, country， 数据有缺失
Age：有缺失，缺失值用NULL表示
2、 book "ISBN";"Book-Title";"Book-Author";"Year-Of-Publication";"Publisher";"Image-URL-S";"Image-URL-M";"Image-URL-L"
ISBN：唯一标识符
Book-Title: 
Book-Author： 
Year-Of-Publication
Publisher
Image-URL-S: 封面小图地址 
Image-URL-M: 封面中图地址
Image-URL-L: 封面大图地址
更深入的处理方法是将图片做目标检测，top-5取出图片中的涉及的事物，或者用看图说话的方式，提取图片中的事物描述
3、 rating "User-ID";"ISBN";"Book-Rating"
用文本卷积的方法，可以对书名进行处理



数值型数据：
用户名、ISBN，

连续型数据：
Age（缺失值统一处理成-1），

类别型数据：Year-Of-Publication
Location：拆分成City，State，Country分别转换成类型型数据
Publisher
Book-Author

Embedding：
Book-Title: 
```