import os
import re
import shutil

# 失败文件列表（请补全所有失败文件名）
failed_files = [
    "大龙虾.txt", "绝胜一击.txt", "洛拉克斯.txt", "低俗小说.txt", "斯坦与奥利.txt", "远大前程.txt", "婚礼歌手.txt",
    "个人日记.txt", "远离天堂.txt", "街头之王.txt", "黑夜传说.txt", "当哈利遇见莎莉.txt", "圣诞故事.txt", "最后的机会.txt",
    "天际线.txt", "夺命追踪.txt", "无法追踪.txt", "博物馆之夜：史密森尼之战.txt", "蒂凡尼的早餐.txt", "九十年代初.txt",
    "幸福.txt", "火焰之中.txt", "索尔之子.txt", "预兆.txt", "史酷比.txt", "第九道门.txt", "木乃伊：龙帝之墓.txt",
    "罗马.txt", "德米安：预言 II.txt", "寄生虫.txt", "当幸福来敲门.txt", "基地.txt", "回到猩球.txt", "隐形斗篷与匕首.txt",
    "有点搞笑的故事.txt", "父亲的遗产.txt", "翻译风波.txt", "比尔和泰德的优异冒险.txt", "终结者3：机器的崛起.txt", "证据.txt",
    "热血警探.txt", "离开她.txt", "佐罗的面具.txt", "陌生人.txt", "露西在天空.txt", "股权.txt", "危险的方法.txt", "领主.txt",
    "艺术家.txt", "山中的小屋.txt", "潜伏2.txt", "女王的游戏.txt", "性别之争.txt", "翻转人生.txt", "爱与友情之间.txt",
    "时间旅行者的妻子.txt", "债务.txt", "1917.txt", "时间机器.txt", "圣痕.txt", "鬼入侵.txt", "赤裸特工3：最终冒犯.txt",
    "美好之地：珍妮特.txt", "另一年.txt", "狮子王.txt", "飞行家.txt", "致命装扮.txt", "四根羽毛.txt", "失窃的夏天.txt",
    "两把枪、一颗手榴弹和一个披萨家伙.txt", "朗格伊公爵夫人.txt", "极其邪恶、极其丑陋.txt", "链式反应.txt", "美国超人.txt",
    "气球飞行员.txt", "肮脏女孩.txt", "婚礼之后.txt", "忘掉莎拉·马歇尔.txt", "第一次分离.txt", "K-PAX.txt", "血色将至.txt",
    "双重身份.txt", "保姆.txt", "前后.txt", "经典电影名.txt", "美好的一年.txt", "超重飞行.txt", "干涉者.txt", "勇敢者游戏.txt",
    "海德公园的阳光.txt", "爱尔兰人.txt", "一个难忘的夜晚.txt", "玛丽菲森特.txt", "边缘的明信片.txt", "玉米地的小孩.txt",
    "原子 блондинка.txt", "狂犬病.txt", "司机.txt", "圣域.txt", "热情骑士.txt", "花木兰.txt", "公司人.txt", "守卫.txt",
    "骗术2.txt", "英格丽的西行.txt", "从这里到虚弱.txt", "明日边缘.txt", "坏月亮.txt", "悠闲的追寻者.txt", "午夜狂奔.txt",
    "怪奇物语.txt", "单身骑士.txt", "野性.txt", "十七岁的边缘.txt", "假期.txt", "神秘河.txt", "蜘蛛侠：平行宇宙.txt",
    "罗宾汉.txt", "无声的证言.txt", "惊声尖叫4.txt", "遗愿清单.txt", "惊天魔盗团.txt", "罗密欧与朱丽叶.txt", "小狗的春天.txt",
    "四十岁的处男.txt", "如果我有一把锤子.txt", "我的希腊婚礼2.txt", "蜘蛛女之吻.txt", "地球停留之日.txt", "你需要的只是爱.txt",
    "乔伊.txt", "林中小屋.txt", "黑色圣诞节.txt", "伊丽莎白镇.txt", "邮差.txt", "阳光下的邪恶.txt", "小屁孩日记.txt",
    "我爱你那么久.txt", "我的爱在天边.txt", "香肠派对.txt", "时尚先锋.txt", "宠物的秘密生活.txt", "小屋惊魂（家庭友好版）.txt",
    "饥饿游戏.txt", "情人节.txt"
]

batch = "batch_000"
base_script_dir = f"/opt/rag_milvus_kb_project/kb_data/script/juben_cn/{batch}"
result_dir = f"/opt/rag_milvus_kb_project/kb_data/scene_result/{batch}"
fail_txt_dir = os.path.join(result_dir, "fail_txt")
os.makedirs(fail_txt_dir, exist_ok=True)

def zh_punct_to_en(s):
    table = {
        '，': ',', '。': '.', '！': '!', '？': '?', '：': ':', '；': ';',
        '（': '(', '）': ')', '【': '[', '】': ']', '“': '"', '”': '"',
        '‘': "'", '’': "'", '、': ',', '《': '<', '》': '>', '——': '-', '…': '...'
    }
    for zh, en in table.items():
        s = s.replace(zh, en)
    return s

def normalize_filename(filename):
    name, ext = os.path.splitext(filename)
    name = name.replace(' ', '_')
    name = zh_punct_to_en(name)
    name = re.sub(r'[^ -\w\-.()]', '', name)  # 只保留常规字符和非ASCII
    return name + ext

def try_delete_result_files(result_dir, name_wo_ext):
    targets = [
        ("scene_name_json", ".json"),
        ("scene_summary_json", ".json"),
        ("scene_summary_xlsx", ".xlsx")
    ]
    for subdir, ext in targets:
        target_dir = os.path.join(result_dir, subdir)
        if not os.path.exists(target_dir):
            continue
        candidates = [
            name_wo_ext,
            os.path.splitext(normalize_filename(name_wo_ext))[0]
        ]
        for cand in set(candidates):
            for file in os.listdir(target_dir):
                if file.startswith(cand):
                    os.remove(os.path.join(target_dir, file))
                    print(f"已删除: {os.path.join(target_dir, file)}")

def copy_failed_txt(base_script_dir, fail_txt_dir, fname):
    src_txt = os.path.join(base_script_dir, fname)
    if os.path.exists(src_txt):
        shutil.copy(src_txt, os.path.join(fail_txt_dir, fname))
        print(f"已复制: {src_txt} -> {fail_txt_dir}")
    else:
        # 尝试规范化名
        norm_fname = normalize_filename(fname)
        src_txt2 = os.path.join(base_script_dir, norm_fname)
        if os.path.exists(src_txt2):
            shutil.copy(src_txt2, os.path.join(fail_txt_dir, norm_fname))
            print(f"已复制: {src_txt2} -> {fail_txt_dir}")
        else:
            print(f"未找到: {src_txt} 或 {src_txt2}")

if __name__ == "__main__":
    for fname in failed_files:
        name_wo_ext = os.path.splitext(fname)[0]
        try_delete_result_files(result_dir, name_wo_ext)
        copy_failed_txt(base_script_dir, fail_txt_dir, fname) 