import os
import time
import random
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
import cv2
import numpy as np
from pywinauto.findwindows import find_window
from skimage.metrics import structural_similarity as ssim

"""
运行本脚本前先使用Reqable等抓包软件

本脚本将在搜狗微信文章搜索指定关键字，并依次访问每个结果
在浏览每篇文章时，通过多次截图对比前后截图相似度来判断图片是否有完全加载完成，加载完成后才会浏览下一篇文章
"""


user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0"
]
window_class_name = "MozillaWindowClass"



def compare_images(img1, img2, threshold=0.998):
    """比较两张图片的相似度"""
    if img1 is None or img2 is None:
        return False

    try:
        # 确保图片尺寸相同
        if img1.shape != img2.shape:
            print("图片尺寸不同，调整大小...")
            height, width = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
            img1 = cv2.resize(img1, (width, height))
            img2 = cv2.resize(img2, (width, height))

        # 转为灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(gray1, gray2, full=True)
        print(f"截图结构相似度得分: {score:.3f}")
        return round(score,3) >= threshold
    except Exception as e:
        print(f"图片比较失败: {e}")
        return False


def capture_screenshot_fallback(driver):
    """使用Selenium截图"""
    try:

        # 使用Selenium内置截图
        screenshot_data = driver.get_screenshot_as_png()

        # 转换为numpy数组
        nparr = np.frombuffer(screenshot_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)


        return img
    except Exception as e:
        print(f"Selenium截图失败: {e}")
        return None




def wait_until_real_article(driver, timeout=15):
    """等待跳转到真实文章页"""
    try:
        # 等待URL包含微信公众号域名
        WebDriverWait(driver, timeout).until(EC.url_contains("mp.weixin.qq.com"))
        print("成功跳转到真实文章页")

        # 额外等待页面内容加载
        time.sleep(3)

        # 等待页面主要内容加载
        try:
            WebDriverWait(driver, 10).until(
                EC.any_of(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "#js_content")),
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".rich_media_content")),
                    EC.presence_of_element_located((By.CSS_SELECTOR, "[data-role='outer']"))
                )
            )
            print("页面内容加载完成")
        except:
            print("无法确认页面内容是否完全加载")

        return True
    except TimeoutException:
        print("页面未跳转到真实文章页，可能被拦截")
        return False


def create_driver_with_random_ua():
    """创建Firefox驱动"""
    firefox_options = Options()
    firefox_options.set_preference("layout.css.devPixelsPerPx", "1.2")  # 缩放比例为 80%

    # 关键配置：启用GPU渲染和硬件加速
    firefox_options.set_preference("layers.acceleration.force-enabled", True)
    firefox_options.set_preference("webgl.force-enabled", True)
    firefox_options.set_preference("media.hardware-video-decoding.force-enabled", True)

    # 设置随机User-Agent
    random_ua = random.choice(user_agents)
    firefox_options.set_preference("general.useragent.override", random_ua)

    # 反检测配置
    firefox_options.set_preference("dom.webdriver.enabled", False)
    firefox_options.set_preference("useAutomationExtension", False)


    try:
        driver = webdriver.Firefox(options=firefox_options)

        # 设置窗口大小
        driver.set_window_size(222, 3333)

        # 反检测脚本
        driver.execute_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
            Object.defineProperty(navigator, 'languages', {get: () => ['zh-CN', 'zh']});
        """)

        print(f"Firefox驱动创建成功，User-Agent: {random_ua}")
        return driver
    except Exception as e:
        print(f"Firefox驱动创建失败: {e}")
        return None


def main():
    """主函数"""
    # 创建目录
    os.makedirs("article_images3", exist_ok=True)

    driver = create_driver_with_random_ua()
    if not driver:
        print("无法创建浏览器驱动，程序退出")
        return

    try:
        for page in range(1,11):

            search_url = f"https://weixin.sogou.com/weixin?type=2&p=&page={page}&query=华南理工大学%E5%B3%BB%E5%BE%B7%E4%B9%A6%E9%99%A2"
            print(f"\n正在访问第 {page} 页: {search_url}")

            try:

                driver.delete_all_cookies()
                driver.get(search_url)
                time.sleep(2)  # 增加等待时间
            except WebDriverException as e:
                print(f"访问搜索页失败: {e}")
                continue

            # 查找文章链接
            title_links = []
            for i in range(10):
                try:
                    link = driver.find_element(By.CSS_SELECTOR, f"a[uigs='article_title_{i}']")
                    title_links.append(link)
                except:
                    pass

            # 备用方法查找链接
            if len(title_links) < 10:
                print("使用备用方法查找文章链接...")
                all_links = driver.find_elements(By.CSS_SELECTOR, "a[uigs^='article_title_']")
                existing_uigs = [link.get_attribute("uigs") for link in title_links]
                for link in all_links:
                    if link.get_attribute("uigs") not in existing_uigs:
                        title_links.append(link)

            urls = [a.get_attribute("href") for a in title_links if a.get_attribute("href")]
            print(f"第 {page} 页找到 {len(urls)} 篇文章")

            for idx, sogou_link in enumerate(urls, 1):
                count_shot=idx
                print(f"\n处理第{idx}篇文章...")
                print(f"   临时链接: {sogou_link}")


                driver.get(sogou_link)

                if not wait_until_real_article(driver):
                    print("跳过此文章")
                    continue

                # 获取真实URL
                real_url = driver.current_url
                print(f"真实链接: {real_url}")

                # 设置页面缩放
                driver.execute_script("document.body.style.zoom='15%'")
                time.sleep(1)

                # 尝试获取窗口句柄
                hwnd = find_window(class_name=window_class_name)
                while True:
                    # 获取页面的 readyState
                    ready_state = driver.execute_script("return document.readyState;")
                    if ready_state == "complete":
                        print("页面加载完成")
                        break
                    time.sleep(1)  # 每秒检查一次
                while True:
                # 第一次截图
                    count_shot = count_shot + 1
                    img_before = capture_screenshot_fallback(driver)
                    if img_before is not None:
                        #cv2.imwrite(f"tiaosi/page{page}_article{idx}_before.png", img_before)
                        print("第一次截图完成")


                    # 等待间隔
                    time.sleep(2)

                    # 第二次截图
                    img_after = capture_screenshot_fallback(driver)

                    if img_after is not None:
                        #cv2.imwrite(f"tiaosi/page{page}_article{idx}_after.png", img_after)
                        print("第二次截图完成")

                    # 比较截图
                    if compare_images(img_before, img_after):
                        print("页面内容稳定")
                        break
                    else:
                        print(f"页面内容发生变化，等待图片加载次数：{count_shot-idx}")
                        if count_shot-idx>5:
                            break



    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"\n程序执行出错: {e}")
    finally:
        try:
            driver.quit()
            print("\n浏览器已关闭，程序结束")
        except:
            pass


if __name__ == "__main__":
    main()
