# ユーザー確認リスト

開発側（監督役）からお願いしている確認をまとめたファイルです。チャットでは流れてしまうので、確認のお願いは今後ここに書き足します。
感想や結果は、各項目の「結果・感想」の欄にそのまま書き込んでください（監督役がこのファイルを読みます）。

最終更新: 2026-10-03

---

## 0. アプリの起動のしかた（共通）

1. エクスプローラーで `C:\mine\stem\scripts\start.bat` をダブルクリックする（または PowerShell で `cd C:\mine\stem; uv run stemapp serve`）。
2. ブラウザで http://127.0.0.1:8000 を開く。
3. 止めるときは黒い画面を閉じる（または Ctrl+C）。

---

## 1. 分け方の聴き比べ（曲「イガク - 重音テト」）　【済】

同じ曲を 7 通りの分け方で分割してあります。どれが良いかを聴いて決めてください。選ばれたものを「標準」にします。

### アプリで聴く（おすすめ）
1. ライブラリで「イガク - 重音テト」の「再生」を押す。
2. 画面上部（曲名の下あたり）の **「分け方」** の選択欄で、下の表の名前を切り替える。再生位置と stem の選択はそのまま保たれます。
3. 聴くとよい組み合わせ:
   - **メインボーカル（キー 2）だけ**、**サブボーカル（キー 3）だけ** をソロにして比べる（メインに入るべき声がサブに入っていないか、サブが他に残っていないか）。
   - **その他（キー 8）だけ** をソロにして、ボーカルの声が混ざっていないか。
   - 組み合わせ「カラオケ（伴奏）」で、声が残っていないか。

### ファイルで聴く
各分け方の stem（FLAC）は次のフォルダにあります。`C:\mine\stem\data\stems\イガク - 重音テト\` の下です。

| アプリでの名前（分け方） | フォルダ | 中身の違い |
| --- | --- | --- |
| 標準 | `standard` | 今の既定 |
| SW のみ（対照） | `exp_resid_split` | ボーカル用モデルの平均をやめたもの |
| ボーカル＝元の曲−楽器 | `exp_resid_vocals` | 「その他」にボーカルが流れ込まない |
| カラオケを元の曲に | `exp_kara_mix` | メイン／サブの分け方を変えたもの |
| カラオケ2種（anvuew＋frazer） | `exp_kara_anvuew` | メイン／サブ用のモデルを2つ平均 |
| ボーカル＝元の曲−楽器＋カラオケ2種を元の曲に | `exp_combo` | 上の組み合わせ |
| ボーカル＝元の曲−楽器＋カラオケ3種を元の曲に | `exp_combo_gabox` | さらに1つ足したもの |

各フォルダの中の `lead_vocal.flac`（メイン）、`backing_vocal.flac`（サブ）、`other.flac`（その他）などを聴き比べてください。

### 結果・感想
- 一番良かった分け方:　4,8 が良かった。他はインストが残ってた。
- 監督役の受け取り（2026-10-03）: 「4」「8」はアプリの分け方の欄の #4（標準 `standard`）と #8（カラオケ2種 `exp_kara_anvuew`）と解釈。メインとサブの分け方が一番正確な **標準（#4）をそのまま既定にする**。#8 は楽器の残りは少ないがボーカルがほぼ全部サブに入るので採用しない。解釈が違えば教えてください。
- 気になった点:メインとサブは4が一番間違えていなかったけど、最後の方と途中はメインが全部サブに入ってることがあった。まあこれは私が手元で編集するかたちで許容してもいいかな。8はほぼ全てがサブに入っていた。

---

## 2. 詳細分割（「もっと分ける」）を聴く　【済】

1. 好きな曲をプレイヤーで開く。
2. stem ボタンの右下にある小さな「分ける」アイコンを押し、方法を選ぶ。
   - ドラム → キック・スネア・タム・ハイハット・ライド・クラッシュ
   - メインボーカル／サブボーカル → 男声・女声、または息
   - その他 → 持続音（パッド等）・短い音（ヒット等）（HPSS）、または Mega 53（シンセ・ストリングス・ブラス・木管・パーカッション）
3. 終わると子の stem のボタンが出ます。ソロにして分かれ方を聴く。「戻す」アイコンで分ける前に戻せます。
4. 分けた stem のファイルは、その曲・分け方のフォルダの下の `drums\`、`other\` などにあります（「保存フォルダを開く」ボタンで開けます）。

聴いてほしい点: パッドとオーケストラヒットがどこに入るか、「残り」に何が入っているか、キック・スネアの分かれ方。

### 結果・感想
-　キックスネアは割といい。その他の分割Mega 53では、オケヒは残り（その他）に入った。パッドはシンセに入った。持続音で分ける手法では短い音の方にはノイズのようなものだけが残った。あまり使えないかも。
- 監督役の対応（2026-10-03）: 「その他」を分ける方法は Mega 53 を既定・先頭にし、HPSS（持続音／短い音）は「実験」として目立たない位置に下げる（T17）。

---

## 3. iPhone での診断　【済】

iPhone 向けの再生（T06b）の作り方を決めるために、iPhone のブラウザで何が使えるかを調べます。

1. PC の `C:\mine\stem\.env` に、次の2行を書き足す（パスコードは好きな文字列）。
   ```
   STEMAPP_ALLOWED_HOSTS=unagi.tail8b25a2.ts.net
   STEMAPP_PASSCODE=好きなパスコード
   ```
2. アプリを起動し直す（0 の手順）。
3. PC の PowerShell で `cd C:\mine\stem; .\scripts\tailscale-serve.ps1 start` を実行する（自分の Tailscale の機器だけに公開。インターネット全体には出ません）。
4. iPhone で Tailscale をオンにし、Safari で `https://unagi.tail8b25a2.ts.net/` を開いてログインする。
5. ライブラリのいちばん下の小さな「端末の診断」を開き、「診断を始める」を押す。「サーバーに保存しました」と出れば完了。
6. （任意）「Web Audio で鳴らす」を押して画面をロックし、10 秒後に解除して結果を選ぶ。「<audio> 要素で鳴らす」でも同じことをする。
7. Safari の共有ボタンから「ホーム画面に追加」し、ホーム画面のアイコンから開いてもう一度 5 を行う。
8. 公開をやめるときは PC で `.\scripts\tailscale-serve.ps1 stop`。

結果はサーバーに自動で保存されるので、ここには「やった」とだけ書けば足ります。

### 3 の手順で止まった件（2026-10-03）→ 直しました
原因: スクリプト `tailscale-serve.ps1` が、PowerShell の文字コード（cp932）のせいで Tailscale の表示名（日本語）を読み違えていました。T17 で直したので、**上の手順 3 のスクリプトがそのまま使えます**（直す前の版で同じ失敗を再現し、直した後に通ることを確認済み）。
もしまた失敗したら、スクリプトの代わりに次の1行でも同じことができます（公開範囲は自分の Tailscale の機器だけ）。
```
& "C:\Program Files\Tailscale\tailscale.exe" serve --bg --https=443 http://127.0.0.1:8000
```

### 結果（2026-10-03、監督役のまとめ）
- ユーザーの報告: Web Audio は消音モードだと画面をオンにしていても鳴らなかった。`<audio>` は消音モードでも鳴った。
- 診断の記録（iPhone の Chrome、iOS 26.6.2）: 音声の形式は Opus・AAC・MP3・FLAC・WAV すべて読み込めた。Web Audio はロックで止まり、`<audio>` はロック中も鳴り続けた。ロック画面の操作（Media Session）は使える。通知（Web Push）は今の開き方では使えない。
- 対応: 今の再生の仕組み（Web Audio）は iPhone の消音モードで鳴らないので、対策を3通り試す実験を診断ページに足す（T06b-0）。できたら下の 5 章でお願いします。

### 結果・感想
- 3ができなかった。以下が実行結果。
PS C:\Users\maeba> cd C:\mine\stem; .\scripts\tailscale-serve.ps1 start
[NG]   ':' または '}' ではなく無効なオブジェクトが渡されました。 (5556): {
  "Version": "1.102.4-t3caf7d9e7-g084ee3b64",
  "TUN": true,
  "BackendState": "Running",
  "HaveNodeKey": true,
  "AuthURL": "",
  "TailscaleIPs": [
    "100.100.95.89",
    "fd7a:115c:a1e0::a028:5f5a"
  ],
  "Self": {
    "ID": "njJAcjmaw521CNTRL",
    "NodeID": 8060427840269600,
    "PublicKey": "nodekey:738bedf4b59c925d967821d958246eb3e2480b1feaf09b6a6d26f443a0a45f59",
    "HostName": "unagi",
    "DNSName": "unagi.tail8b25a2.ts.net.",
    "OS": "windows",
    "UserID": 7475193010315373,
    "TailscaleIPs": [
      "100.100.95.89",
      "fd7a:115c:a1e0::a028:5f5a"
    ],
    "AllowedIPs": [
      "100.100.95.89/32",
      "fd7a:115c:a1e0::a028:5f5a/128"
    ],
    "Addrs": [
      "123.226.19.113:41641",
      "123.226.19.113:11043",
      "172.16.30.208:41641"
    ],
    "CurAddr": "",
    "Relay": "tok",
    "PeerRelay": "",
    "RxBytes": 0,
    "TxBytes": 0,
    "Created": "2026-09-25T00:18:00.031552001Z",
    "LastWrite": "0001-01-01T00:00:00Z",
    "LastSeen": "0001-01-01T00:00:00Z",
    "LastHandshake": "0001-01-01T00:00:00Z",
    "Online": true,
    "ExitNode": false,
    "ExitNodeOption": false,
    "Active": false,
    "PeerAPIURL": [
      "http://100.100.95.89:37413",
      "http://[fd7a:115c:a1e0::a028:5f5a]:46907"
    ],
    "TaildropTarget": 0,
    "NoFileSharingReason": "",
    "Capabilities": [
      "HTTPS://TAILSCALE.COM/s/DEPRECATED-NODE-CAPS#see-https://github.com/tailscale/tailscale/issues/11508",
      "default-auto-update",
      "https",
      "https://tailscale.com/cap/file-sharing",
      "https://tailscale.com/cap/is-admin",
      "https://tailscale.com/cap/is-owner",
      "https://tailscale.com/cap/ssh",
      "https://tailscale.com/cap/tailnet-lock",
      "probe-udp-lifetime",
      "ssh-behavior-v1",
      "ssh-env-vars",
      "store-appc-routes",
      "tailnet-display-name"
    ],
    "CapMap": {
      "default-auto-update": [
        true
      ],
      "https": null,
      "https://tailscale.com/cap/file-sharing": null,
      "https://tailscale.com/cap/is-admin": null,
      "https://tailscale.com/cap/is-owner": null,
      "https://tailscale.com/cap/ssh": null,
      "https://tailscale.com/cap/tailnet-lock": null,
      "probe-udp-lifetime": null,
      "ssh-behavior-v1": null,
      "ssh-env-vars": null,
      "store-appc-routes": null,
      "tailnet-display-name": [
        "maebashoten.jp@gmail.com"
      ]
    },
    "InNetworkMap": true,
    "InMagicSock": false,
    "InEngine": false
  },
  "Health": [],
  "MagicDNSSuffix": "tail8b25a2.ts.net",
  "CurrentTailnet": {
    "Name": "maebashoten.jp@gmail.com",
    "MagicDNSSuffix": "tail8b25a2.ts.net",
    "MagicDNSEnabled": true
  },
  "CertDomains": [
    "unagi.tail8b25a2.ts.net"
  ],
  "ExtraRecords": null,
  "Peer": {
    "nodekey:af8d461ac5509559950932cd9058d990185ab8526a5374fc7da443f616ce8d4c": {
      "ID": "nsLdsemMmK11CNTRL",
      "NodeID": 2403077456433456,
      "PublicKey": "nodekey:af8d461ac5509559950932cd9058d990185ab8526a5374fc7da443f616ce8d4c",
      "HostName": "localhost",
      "DNSName": "ipad157.tail8b25a2.ts.net.",
      "OS": "iOS",
      "UserID": 7475193010315373,
      "TailscaleIPs": [
        "100.78.141.76",
        "fd7a:115c:a1e0::1628:8d4d"
      ],
      "AllowedIPs": [
        "100.78.141.76/32",
        "fd7a:115c:a1e0::1628:8d4d/128"
      ],
      "Addrs": null,
      "CurAddr": "",
      "Relay": "tok",
      "PeerRelay": "",
      "RxBytes": 0,
      "TxBytes": 0,
      "Created": "2026-09-29T04:07:50.42535893Z",
      "LastWrite": "0001-01-01T00:00:00Z",
      "LastSeen": "0001-01-01T00:00:00Z",
      "LastHandshake": "0001-01-01T00:00:00Z",
      "Online": true,
      "ExitNode": false,
      "ExitNodeOption": false,
      "Active": false,
      "PeerAPIURL": [
        "http://100.78.141.76:37477",
        "http://[fd7a:115c:a1e0::1628:8d4d]:38241"
      ],
      "TaildropTarget": 1,
      "NoFileSharingReason": "",
      "InNetworkMap": true,
      "InMagicSock": true,
      "InEngine": false,
      "KeyExpiry": "2027-03-28T04:07:50Z"
    },
    "nodekey:f9387c3490a870bceca7aae077c36ec3f3a22605e412e2bd655c89f983b81b31": {
      "ID": "nc8sekGy8C21CNTRL",
      "NodeID": 8853950713847065,
      "PublicKey": "nodekey:f9387c3490a870bceca7aae077c36ec3f3a22605e412e2bd655c89f983b81b31",
      "HostName": "localhost",
      "DNSName": "iphone-13.tail8b25a2.ts.net.",
      "OS": "iOS",
      "UserID": 7475193010315373,
      "TailscaleIPs": [
        "100.120.27.49",
        "fd7a:115c:a1e0::8328:1b32"
      ],
      "AllowedIPs": [
        "100.120.27.49/32",
        "fd7a:115c:a1e0::8328:1b32/128"
      ],
      "Addrs": null,
      "CurAddr": "172.16.31.147:41641",
      "Relay": "tok",
      "PeerRelay": "",
      "RxBytes": 1012,
      "TxBytes": 180,
      "Created": "2026-09-25T00:20:33.983513221Z",
      "LastWrite": "2026-10-03T11:19:28.1204059+09:00",
      "LastSeen": "2026-10-03T01:40:00.1Z",
      "LastHandshake": "2026-10-03T11:19:07.1754464+09:00",
      "Online": true,
      "ExitNode": false,
      "ExitNodeOption": false,
      "Active": true,
      "PeerAPIURL": [
        "http://100.120.27.49:43866",
        "http://[fd7a:115c:a1e0::8328:1b32]:54352"
      ],
      "TaildropTarget": 1,
      "NoFileSharingReason": "",
      "InNetworkMap": true,
      "InMagicSock": true,
      "InEngine": true,
      "KeyExpiry": "2027-03-24T00:20:33Z"
    }
  },
  "User": {
    "7475193010315373": {
      "ID": 7475193010315373,
      "LoginName": "maebashoten.jp@gmail.com",
      "DisplayName": "蜑肴ｭｯ蝠・ｺ・,
      "ProfilePicURL": "https://lh3.googleusercontent.com/a/ACg8ocLhpnCrju2Ic9hLWlh-SWgFOjKJodzqAz7UIdMlBp1Zb4-Pjvc=s96-c"
    }
  },
  "ClientVersion": null
}
PS C:\mine\stem>

---

## 4. 速度変更　【済】

- 2026-10-03 の感想: 「ピッチを変える」「ピッチを変えない（すぐ）」はとても良い。
- 追加で気づいたことがあれば:
  - 「すぐ」の方式では音が約 0.12 秒遅れて出ます（stem の ON/OFF も同じだけ遅れる）。気になるか:
  - 「高音質（サーバーで作る）」との音質の差:

---

## 5. iPhone 再生の実験（消音モード・ロック中）　【未】

診断で「Web Audio（今の再生の仕組み）は消音モードで鳴らない・ロックで止まる」と分かったので、対策を 3 通り試します。結果で iPhone 向けの再生（T06b）の作り方を決めます。

準備（3 章と同じ）:
1. PC でアプリを起動し直す（新しい版を読み込むため。0 の手順）。
2. PC の PowerShell で `cd C:\mine\stem; .\scripts	ailscale-serve.ps1 start`。
3. iPhone で `https://unagi.tail8b25a2.ts.net/` を開き、ライブラリのいちばん下の「端末の診断」を開く。
4. 「iPhone 再生の実験（消音モード・ロック中）」の欄までスクロールする。

各実験（A・B・C・D の順に、1 つずつ）:
1. **本体の消音スイッチを ON（消音）にする。**
2. 実験のボタン（「A を鳴らす」など）を押す。メロディが鳴るはずです。
3. 「消音モードで」の質問に答える（鳴った／鳴らなかった）。
4. 画面をロックして 10 秒待つ。ロック画面に曲名「stemapp 実験 X」と再生ボタンが出ているかも見る。
5. ロックを解除して、「ロック中」「ロック画面」の質問に答え、「記録する」を押す。
   - A は端末によっては「この端末には無い」と出て自動で記録されます。その場合は次へ。

| 実験 | 中身 |
| --- | --- |
| A | ブラウザに「音楽再生用です」と伝えてから Web Audio で鳴らす |
| B | 無音の `<audio>` を一緒に流しながら Web Audio で鳴らす |
| C | Web Audio の音を `<audio>` から出す |
| D | `<audio>` だけ（比較用。前回は消音でも鳴り、ロック中も続いた） |

結果はサーバーに保存されるので、ここには「やった」と、気づいたことがあれば書いてください。終わったら `.\scripts	ailscale-serve.ps1 stop` で公開をやめられます。

### 結果・感想
-

---

## 済んだこと（記録）

- 2026-10-03: 既存の曲 19 件を新しい名前のフォルダ（`data\stems\<元のファイル名>\<分け方>\`）へ移した。名前の末尾の `[sm…]` などの ID はそのまま残した（取りたい場合は言ってください）。
- 2026-10-03: 「アカ通信ン」の stem を `C:\works\クールポコ\結果\アカーン\` から元の場所へコピーして戻した（`C:\works\` 側はそのまま）。その後の移行で `data\stems\アカ通信ン [sm46822928]\standard\` に移っている。
- 2026-10-03: Mega 53 の重み（ライセンス表示なし）を個人利用の前提で使うことを了承済み。
- 2026-10-03: DB の作り直し（番号の使い回し防止）を実施。バックアップは `data\backup\` にある。
