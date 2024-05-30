import numpy as np
import sys
import math
import cv2

beam_altitude_angles = [
        46.84,
        45.81,
        45.09,
        44.66,
        43.82,
        42.85,
        42.13,
        41.68,
        40.87,
        39.89,
        39.19,
        38.72,
        37.91,
        36.96,
        36.24,
        35.77,
        34.97,
        34.03,
        33.32,
        32.84,
        32.04,
        31.12,
        30.42,
        29.92,
        29.12,
        28.22,
        27.52,
        27.01,
        26.22,
        25.34,
        24.64,
        24.12,
        23.34,
        22.47,
        21.78,
        21.23,
        20.44,
        19.61,
        18.92,
        18.35,
        17.57,
        16.77,
        16.07,
        15.49,
        14.71,
        13.93,
        13.25,
        12.64,
        11.87,
        11.11,
        10.42,
        9.79,
        9.04,
        8.29,
        7.6,
        6.96,
        6.2,
        5.47,
        4.79,
        4.12,
        3.37,
        2.66,
        1.97,
        1.28,
        0.54,
        -0.15,
        -0.85,
        -1.55,
        -2.31,
        -2.97,
        -3.66,
        -4.39,
        -5.14,
        -5.79,
        -6.48,
        -7.24,
        -7.98,
        -8.62,
        -9.31,
        -10.09,
        -10.84,
        -11.45,
        -12.15,
        -12.95,
        -13.69,
        -14.29,
        -14.98,
        -15.81,
        -16.55,
        -17.14,
        -17.84,
        -18.69,
        -19.44,
        -20,
        -20.71,
        -21.57,
        -22.33,
        -22.87,
        -23.58,
        -24.48,
        -25.23,
        -25.76,
        -26.48,
        -27.39,
        -28.14,
        -28.67,
        -29.39,
        -30.33,
        -31.08,
        -31.59,
        -32.33,
        -33.29,
        -34.03,
        -34.53,
        -35.28,
        -36.27,
        -37,
        -37.5,
        -38.25,
        -39.27,
        -40.01,
        -40.49,
        -41.25,
        -42.3,
        -43.04,
        -43.52,
        -44.28,
        -45.35
    ]
pixel_shift_by_row = [
            32,
            11,
            -10,
            -31,
            31,
            10,
            -10,
            -29,
            29,
            10,
            -9,
            -29,
            29,
            10,
            -9,
            -28,
            28,
            9,
            -9,
            -27,
            27,
            9,
            -9,
            -26,
            26,
            9,
            -9,
            -26,
            26,
            9,
            -8,
            -25,
            26,
            9,
            -8,
            -25,
            25,
            8,
            -8,
            -25,
            25,
            8,
            -8,
            -24,
            25,
            8,
            -8,
            -24,
            24,
            8,
            -8,
            -24,
            24,
            8,
            -8,
            -24,
            24,
            8,
            -8,
            -24,
            24,
            8,
            -8,
            -24,
            24,
            8,
            -8,
            -24,
            24,
            8,
            -8,
            -24,
            24,
            8,
            -8,
            -24,
            24,
            8,
            -8,
            -24,
            24,
            8,
            -8,
            -24,
            24,
            8,
            -8,
            -25,
            25,
            8,
            -8,
            -25,
            25,
            8,
            -8,
            -25,
            25,
            8,
            -9,
            -26,
            26,
            9,
            -9,
            -26,
            26,
            9,
            -9,
            -27,
            27,
            9,
            -9,
            -27,
            27,
            9,
            -9,
            -28,
            28,
            9,
            -10,
            -29,
            29,
            10,
            -10,
            -30,
            30,
            10,
            -10,
            -31
        ]


def similar_ranges(r, i):
    sim_num=0
    rr = r[i]
    xx = i%1024
    yy = i//1024
    if rr!=0:
      for x in range(-4,5):
         for y in range(-2,3):
            if y+yy>=0 and y+yy<128:
              r2 = r[((x+xx)%1024)+(y+yy)*1024]
              if abs(r2-rr)<200:
                  sim_num+=1
    return sim_num>5
    

def main():
    # cloud = pcl.load_XYZRGB(
    #     './examples/pcldata/tutorials/table_scene_mug_stereo_textured.pcd')
    #cloud = pcl.load(sys.argv[1])
    f = open(sys.argv[1])
    ranges = [0]*(1024*128)
    img = np.zeros((128,1024,3), np.uint8)
    for i in range(11):
        f.readline()
        
    line = 0
    for i in range(1024*128):
        row = i//1024
        l=f.readline().split()
        ranges[i]=int(l[8])
        gray = ranges[i]//100
        if gray>255:
          gray=255
        if ranges[i]>0:
          xp = float(l[0])
          yp = float(l[1])
          zp = float(l[2])
          dist = (xp*xp+yp*yp+zp*zp)**0.5
          alt = math.asin(zp/dist)*180/math.pi
          
          while len(beam_altitude_angles)>line+1 and abs(alt-beam_altitude_angles[line])>abs(alt-beam_altitude_angles[line+1]):
            line+=1
          
          azi = math.atan2(yp,xp)*180/math.pi
          azi_step = 512 - int(math.atan2(yp,xp)*512/math.pi+0.5)
          print(i, alt, line, row, azi, azi_step, (i+pixel_shift_by_row[row])%1024, azi_step-((i+pixel_shift_by_row[row])%1024))
          x = (i+pixel_shift_by_row[row])%1024
          y = row
          img[y][x][0]=gray
          img[y][x][1]=gray
          img[y][x][2]=gray
          
    #print(ranges)
    cv2.imshow("ranges",img)
    cv2.waitKey(0)


if __name__ == "__main__":
    # import cProfile
    # cProfile.run('main()', sort='time')
    main()
