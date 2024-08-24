---
title: Lane Detect
subtitle: basic image process
date: 2021-02-22
tags: ["c++", "image process", "lane detect"]
---





使用数字图像的基础方法进行车道线检测（概率霍夫+聚类），不调用opencv相关算法，在**TuSimple Lane Dataset**评测。

<!--more-->

数字图像处理课程的大作业，要求c++实现，不能使用神经网络，除输入输出外不调用opencv库。

![](/media/lane/src.png)

![](/media/lane/res.png)

### 程序思路

- 转换到HSL空间， 提取黄色车道线

- 转换为灰度图并进行对比度拉伸（Gamma）, 设阈值得到白色车道线

- 相加两张图片，并遮盖天空（固定区域）

- 通过概率霍夫检测直线

- 对直线进行聚类


#### 1.提取车道线

- 黄色：

  转换到HSL色彩空间，取H：0\~40，L大于100左右来剔除周围黄土

- 白色：

  对灰度图进行对比度拉伸，然后根据平均亮度取一个阈值（写完仔细一想直方图均衡化就行了，但这样效果感觉够用，时间也比较紧张就没改）

```c++
	// contrast stretching
    Mat image_Gray;
    cvtColor(src, image_Gray, cv::COLOR_BGR2GRAY);
    uchar* ptr = image_Gray.ptr();
    for (int i = 0; i < image_Gray.cols * image_Gray.rows; i++, ptr++) {
        int val = *ptr;
        *ptr = 255 * pow(val / 255.0, 1.2);//gamma
    }

```

根据图片亮度选择阈值，只计算下半张图片以除去天空影响，得到平均亮度，乘以一个系数作为亮度阈值（1.4左右）

```c++
    // determine lightness threshold = average lightness * coefficient
    long threshold_image = 0;
    for (int i = 360; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            threshold_image += image_HSL.at<Vec3b>(i, j)[2];
        }
    }
    threshold_image /= src.cols * (src.rows - 360);
    threshold_image *= 1.2; //1.4

```

原图：

![](/media/lane/src.png)

灰度图：

![](/media/lane/gray.png)

计算阈值并遮盖天空后：
![](/media/lane/white_threshold.png)



### 2.概率霍夫直线检测

一开始写了标准霍夫，发现效果很不好，尤其是因为需要检测出图中所有车道线，所以对照opencv源码写了概率霍夫。概率霍夫不进行边缘检测得到的效果更好，故没有进行边缘检测。

记录图中所有点：

```c++
// collect points
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (image.at<uchar>(i, j) > 0) {
                pts.push_back(Point(i, j));
                msk[i * width + j] = (uchar)1;
            }
            else {
                msk[i * width + j] = 0;
            }
        }
    }

```

随机选择点：

```c++
// select point
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    std::shuffle(pts.begin(), pts.end(), generator); // shuffle points randomly

```



对每个点各角度计算vote(为方便直接固定步长1度):

```c++
for (int n = 0; n < 180; n++) {
            int r = round(j * ttab[2 * n] + i * ttab[2 * n + 1]);
            r += (numrho - 1)/2;
            int vote = ++acc[n*180+r];
            if (max_vote < vote) {
                max_vote = vote;
                max_angle = n;
                rec_r = r- (numrho - 1) / 2;
            }
        }

```

如果当前[rho，theta]的vote未达到阈值，计算下一个点

```c++
if (max_vote < threshold )
            continue;

```

如果达到vote阈值，沿此方向找直线，大于允许的最大间隔长度则停止，期间更新直线的起点和终点。 计算di时由于取dj=1, di很小, 需要扩大移位避免精度问题。

```c++
// go left or right
for (int k = 0; k < 2; k++) {
    int gap = 0;

    int i_mv = i;
    int j_mv = j;

    int dj = 1;
    int di = cos_dir < 0 ? cot_dir : -cot_dir;

    i_mv = i_mv << shift;

    if (k == 1) {
        di = -di;
        dj = -dj;
    }
    for (; i_mv>>shift < height && i_mv>>shift > 0 && j_mv < width && j_mv > 0; j_mv += dj, i_mv += di) {
        if (msk[(i_mv >> shift) * width + j_mv]) {
            gap = 0;
            if (k == 0) {
                line_st.y = i_mv >> shift;
                line_st.x = j_mv;
            }
            else {
                line_end.y = i_mv >> shift;
                line_end.x = j_mv;
            }
        }
        else if (++gap > lineGap)break;
    }
}

```

确定直线后，再对直线上每个点各角度的vote--，消除本条直线的影响。



### 3.直线聚类

对直线进行聚类，因为前一步检测效果还行，聚类的要求不高，随便写了一个简单的分组，由于较长直线一般是最符合实际方向的，将每组中最长的线作为代表直线。

Cluster记录[rho, theta, cnt, length]

```c++
vector<Vec4i> Hough::ClusterLines(std::vector<Vec4i>& lines, vector<Vec2d>& pos, int threshold_rho, int threshold_theta) {
    int siz = lines.size();
    int* vis = new int[siz];
    vector<Vec4i> cluster;
    for (int k = 0; k < siz; k++)vis[k] = 0;

    for (int i = 0; i < siz; i++) {
        if (vis[i])continue;
        int m_theta = pos[i][1];
        int m_rho = pos[i][0];
        int m_len = sqrt(pow(lines[i][0] - lines[i][2], 2) + pow(lines[i][1] - lines[i][3], 2));
        int cnt = 1;
        vis[i] = 1;
        for (int j = 0; j < siz; j++) {
            if (vis[j])continue;
            // find nearby lines and group them
            if (abs(pos[i][0] - pos[j][0]) < threshold_rho && abs(pos[i][1] - pos[j][1]) < threshold_theta) {
                int cl_len = sqrt(pow(lines[j][0] - lines[j][2], 2) + pow(lines[j][1] - lines[j][3], 2));
                vis[j] = 1;
                // take longest line as represent
                if (m_len > cl_len) {
                    m_len = cl_len;
                    m_rho = pos[j][0];
                    m_theta = pos[j][1];
                }
                cnt++;
                
            }
        }
        MergeCluster(cluster, m_rho, m_theta, m_len, cnt, threshold_rho, threshold_theta);
    }

    return cluster;
}

```

这样分组后，可能有实际很近的组因为选到了边缘直线最为初始分组依据被分开，所以递归合并各组`MergeCluster`，对于长度更长的组，合并时给予更大权重。

cluster\[no][rho, theta, cnt, length]

```c++
for (int p = 0; p < cluster.size(); p++) {
    if (abs(cluster[p][0] - rho_mean) < threshold_rho && abs(cluster[p][1] - theta_mean) < threshold_theta) {
        // weighted merge
        int new_rho = (cluster[p][3]*cluster[p][0] + len_mean*rho_mean) / (len_mean+cluster[p][3]);
        int new_theta = (cluster[p][3] * cluster[p][1] + len_mean * theta_mean) / (len_mean + cluster[p][3]);
        int new_cnt = cluster[p][2] + cnt;

        cluster.erase(cluster.begin() + p);
        MergeCluster(cluster, new_rho, new_theta, new_cnt, new_len, threshold_rho, threshold_theta);
    }
}

```

如果合并后数量仍多于4条，不是噪声影响就是有五条车道，噪声的情况不谈，5条车道的话因为要求最多输出4条，选择4条最可能的直线（计算与人为设定的4条标准直线的距离）

结果：

红色为霍夫直线，绿色为聚类后直线

![](/media/lane/res.png)

输出时截掉上半部分直线。

### 4.结果及分析

使用TuSimple提供的测试脚本，绿色为ground truth, 蓝色为预测结果，输出(accuracy, fp, fn)

对于光照条件正常，车道线清晰的情况比较准确：

![](/media/lane/good.png)

![](/media/lane/good2.png)

如果地面没有白色线或间距大，没办法检测出(右侧第二根):


![](/media/lane/no_white.png)

此方法也没有考虑弯道：

![](/media/lane/curve.png)