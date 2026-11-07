from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

# Original, human-authored bilingual defensive-security instructions.
# Each topic contributes an explanation, a practical checklist and a common
# failure-mode example in both languages. The six generation prompts and the
# separate held-out evaluation set are deliberately not copied here.

TOPICS = [
    {
        "en": (
            "multi-factor authentication",
            "Multi-factor authentication adds a second independent proof of identity, such as an authenticator code or security key. A stolen password alone is then less likely to give an attacker access. Prefer phishing-resistant security keys where supported, keep recovery methods protected, and never approve an unexpected sign-in request.",
            "Enable MFA on email, administrator, cloud and financial accounts first. Register two approved recovery methods, store recovery codes offline, test account recovery, and review active sessions after enabling it.",
            "Do not treat an SMS code as a guarantee against every attack. SIM swaps and real-time phishing can defeat some one-time codes. Use a passkey or hardware security key for high-value accounts and reject prompts you did not initiate."
        ),
        "hi": (
            "बहु-कारक प्रमाणीकरण",
            "बहु-कारक प्रमाणीकरण पहचान की पुष्टि के लिए पासवर्ड के साथ एक स्वतंत्र प्रमाण भी माँगता है, जैसे प्रमाणक ऐप का कोड या सुरक्षा कुंजी। इससे केवल चोरी हुआ पासवर्ड पर्याप्त नहीं रहता। जहाँ संभव हो फ़िशिंग-रोधी सुरक्षा कुंजी चुनें, recovery तरीके सुरक्षित रखें और अनजान sign-in अनुरोध स्वीकार न करें।",
            "सबसे पहले ईमेल, administrator, cloud और वित्तीय खातों पर MFA चालू करें। दो सुरक्षित recovery तरीके दर्ज करें, recovery codes को ऑफलाइन रखें, खाते की पुनर्प्राप्ति जाँचें और सक्रिय sessions की समीक्षा करें।",
            "SMS कोड को पूर्ण सुरक्षा न मानें। SIM swap और real-time phishing कुछ one-time codes को चुरा सकते हैं। महत्वपूर्ण खातों के लिए passkey या hardware security key चुनें और अपने द्वारा शुरू न किए गए अनुरोध को अस्वीकार करें।"
        )
    },
    {
        "en": (
            "least-privilege access",
            "The principle of least privilege gives a person or service only the permissions required for its current task. It limits accidental changes and reduces the damage possible after an account is compromised. Permissions should have an owner, a business reason and a review date.",
            "Use separate standard and administrator accounts, grant access through approved groups, set expiry dates for temporary access, log privileged actions, and review membership whenever a role changes.",
            "Avoid giving broad administrator rights for convenience. A temporary task rarely justifies permanent access. Use time-limited elevation, record the approval, and remove the permission when the task ends."
        ),
        "hi": (
            "न्यूनतम आवश्यक अनुमति",
            "Least privilege का अर्थ है कि किसी व्यक्ति या service को केवल उतनी permissions मिलें जितनी उसके वर्तमान काम के लिए आवश्यक हैं। इससे गलती से बदलाव और account compromise होने पर नुकसान सीमित रहता है। हर permission का owner, काम से जुड़ा कारण और review date होना चाहिए।",
            "सामान्य और administrator खातों को अलग रखें, approved groups से access दें, अस्थायी access की expiry तय करें, privileged actions का log रखें और भूमिका बदलने पर membership जाँचें।",
            "सुविधा के लिए सभी को administrator अधिकार न दें। अस्थायी काम के लिए स्थायी अनुमति उचित नहीं है। समय-सीमित access दें, approval दर्ज करें और काम पूरा होते ही अनुमति हटा दें।"
        )
    },
    {
        "en": (
            "phishing messages",
            "Phishing uses a deceptive message to make someone reveal information, open an unsafe attachment or approve an action. Check the actual sender address and destination domain rather than trusting the displayed name or urgent wording. Verify unusual requests through a separate trusted channel.",
            "Do not open unexpected attachments, inspect links before visiting them, report suspicious messages using the approved process, and contact the security team quickly if credentials were entered.",
            "Do not forward a suspicious message widely or reply to challenge the sender. Preserve it for the security team, avoid interacting with its links, and use a known contact method to verify any payment or password request."
        ),
        "hi": (
            "फ़िशिंग संदेश",
            "फ़िशिंग में धोखे वाला संदेश भेजकर किसी व्यक्ति से जानकारी, असुरक्षित attachment खोलने या किसी कार्रवाई की स्वीकृति लेने की कोशिश की जाती है। दिख रहे नाम या जल्दबाज़ी वाले शब्दों पर भरोसा न करें; वास्तविक sender address और destination domain जाँचें। असामान्य अनुरोध को अलग, भरोसेमंद माध्यम से सत्यापित करें।",
            "अनपेक्षित attachment न खोलें, link पर जाने से पहले उसका पता जाँचें, संदिग्ध संदेश को तय प्रक्रिया से report करें और credentials भर दिए हों तो तुरंत security team को बताएँ।",
            "संदिग्ध संदेश को कई लोगों को forward न करें और sender से बहस न करें। उसे security team के लिए सुरक्षित रखें, link पर interaction न करें और payment या password संबंधी अनुरोध की पुष्टि ज्ञात संपर्क माध्यम से करें।"
        )
    },
    {
        "en": (
            "network firewalls",
            "A firewall applies network rules to allow or block traffic between systems or network zones. Good rules are specific about source, destination, port and purpose. A firewall is one layer of defence; it does not replace endpoint protection, secure applications or identity controls.",
            "Document the owner and business reason for each rule, allow only required flows, review temporary exceptions, log denied and unusual traffic, and test changes before broad deployment.",
            "Avoid a permanent allow-all rule to fix a connectivity problem. Identify the required service and narrow the rule to the correct hosts and ports. Set an expiry and verify the change with the application owner."
        ),
        "hi": (
            "नेटवर्क फ़ायरवॉल",
            "फ़ायरवॉल नियमों के आधार पर अलग-अलग systems या network zones के बीच traffic को अनुमति देता है या रोकता है। अच्छे नियम source, destination, port और उद्देश्य को स्पष्ट रखते हैं। फ़ायरवॉल सुरक्षा की एक परत है; यह endpoint protection, सुरक्षित applications या identity controls का विकल्प नहीं है।",
            "हर rule का owner और व्यावसायिक कारण दर्ज करें, केवल आवश्यक traffic अनुमति दें, अस्थायी exceptions की समीक्षा करें, असामान्य traffic log करें और बदलाव को व्यापक रूप से लागू करने से पहले test करें।",
            "कनेक्टिविटी समस्या दूर करने के लिए स्थायी allow-all rule न बनाएँ। आवश्यक service पहचानें और नियम को सही hosts तथा ports तक सीमित करें। expiry तय करें और application owner के साथ जाँच करें।"
        )
    },
    {
        "en": (
            "SQL injection prevention",
            "SQL injection occurs when untrusted input changes the meaning of a database query. Parameterized queries keep data separate from query instructions and are the primary defence. Apply least-privilege database permissions and avoid returning detailed database errors to users.",
            "Use parameterized statements or a safe query builder, validate input for its expected type and length, use a restricted database account, keep libraries updated, and test query handling.",
            "Do not rely on blocking a few suspicious characters. Attackers can represent input in many ways. Keep query structure fixed, pass user values as parameters, and review every place where input reaches a database."
        ),
        "hi": (
            "SQL injection से बचाव",
            "SQL injection तब हो सकता है जब अविश्वसनीय input database query का अर्थ बदल दे। Parameterized queries में data और query instructions अलग रहते हैं; यह मुख्य बचाव है। Database permissions सीमित रखें और उपयोगकर्ता को विस्तृत database errors न दिखाएँ।",
            "Parameterized statements या सुरक्षित query builder उपयोग करें, input का type और length जाँचें, सीमित अधिकार वाला database account रखें, libraries update करें और query handling test करें।",
            "कुछ संदिग्ध characters block करने पर निर्भर न रहें। Input कई रूपों में आ सकता है। Query की संरचना स्थिर रखें, user values को parameters के रूप में दें और database तक पहुँचने वाले हर input path की समीक्षा करें।"
        )
    },
    {
        "en": (
            "TLS and data in transit",
            "TLS protects data while it travels between a client and a server by authenticating the server and encrypting the connection. It does not automatically secure data after it reaches an endpoint. Certificate validation, current protocol settings and safe key handling are essential.",
            "Use current TLS settings, validate certificates, renew certificates before expiry, protect private keys, disable obsolete protocols, and monitor certificate errors.",
            "Do not ignore certificate warnings just to make a connection work. A warning may indicate a wrong host, an expired certificate or interception. Verify the destination and repair the certificate chain."
        ),
        "hi": (
            "TLS और संचार के दौरान data की सुरक्षा",
            "TLS client और server के बीच यात्रा करते समय data को encrypt करता है और server की पहचान की पुष्टि करता है। Endpoint पर पहुँचने के बाद data अपने-आप सुरक्षित नहीं हो जाता। Certificate validation, आधुनिक protocol settings और सुरक्षित key handling आवश्यक हैं।",
            "आधुनिक TLS settings रखें, certificates validate करें, expiry से पहले renew करें, private keys सुरक्षित रखें, पुराने protocols बंद करें और certificate errors monitor करें।",
            "केवल connection चलाने के लिए certificate warning को अनदेखा न करें। यह गलत host, expired certificate या interception का संकेत हो सकता है। Destination सत्यापित करें और certificate chain ठीक करें।"
        )
    },
    {
        "en": (
            "secure password storage",
            "A service should never store readable passwords. Store a password verifier produced by a slow, purpose-built password-hashing function with a unique salt for every account. A modern password-hashing scheme makes large-scale guessing more expensive and should be configured for the available hardware.",
            "Use a maintained password-hashing library, generate a unique random salt per account, choose a current work factor, protect the verifier database, and support password changes after suspected exposure.",
            "Do not use plain SHA-256 or reversible encryption as a password-storage substitute. They are not designed for password guessing resistance. Use a dedicated password-hashing function and review its parameters periodically."
        ),
        "hi": (
            "पासवर्ड को सुरक्षित रूप से रखना",
            "किसी service को readable passwords कभी store नहीं करने चाहिए। हर खाते के लिए अलग salt के साथ slow, purpose-built password-hashing function से बना verifier रखें। आधुनिक password hashing बड़े पैमाने पर guessing को महँगा बनाती है और इसे उपलब्ध hardware के अनुसार configure करना चाहिए।",
            "maintained password-hashing library चुनें, हर खाते के लिए नया random salt बनाएँ, वर्तमान work factor रखें, verifier database सुरक्षित करें और exposure की आशंका पर password बदलने की सुविधा दें।",
            "Password storage के लिए plain SHA-256 या reversible encryption का विकल्प न अपनाएँ। ये password guessing को कठिन बनाने के लिए नहीं बने हैं। Dedicated password-hashing function उपयोग करें और parameters की समय-समय पर समीक्षा करें।"
        )
    },
    {
        "en": (
            "security patch management",
            "Patch management reduces exposure to known software weaknesses. Prioritize updates using severity, exploitability, asset importance and exposure rather than treating every update equally. Test critical updates where practical, but do not let testing become an indefinite reason to delay urgent fixes.",
            "Maintain an asset inventory, monitor vendor advisories, prioritize internet-facing and critical systems, test updates, schedule deployment, and verify that the installed version changed.",
            "Do not assume that a patch was applied because a deployment job reported success. Confirm the version on the device, check failed endpoints, and document any approved exception with an expiry date."
        ),
        "hi": (
            "Security patch management",
            "Patch management software की ज्ञात कमजोरियों से जोखिम कम करता है। हर update को समान न मानें; severity, exploitability, asset की महत्ता और exposure के आधार पर प्राथमिकता दें। जहाँ संभव हो महत्वपूर्ण updates test करें, लेकिन testing को जरूरी fix टालने का अनिश्चित कारण न बनने दें।",
            "Assets की inventory रखें, vendor advisories देखें, internet-facing और महत्वपूर्ण systems को प्राथमिकता दें, updates test करें, deployment schedule बनाएँ और installed version की पुष्टि करें।",
            "केवल deployment job सफल दिखने से यह न मानें कि patch लग गया। Device पर version जाँचें, असफल endpoints पहचानें और approved exception की expiry दर्ज करें।"
        )
    },
    {
        "en": (
            "ransomware resilience",
            "Ransomware resilience depends on preventing initial access, limiting privileges, detecting unusual activity and restoring clean systems. Maintain offline or immutable backups and test restoration. During a suspected incident, use the approved response plan and preserve evidence before making changes.",
            "Keep tested backups, separate backup credentials, patch exposed systems, restrict remote access, monitor mass file changes, and rehearse recovery with service owners.",
            "Do not connect backup storage to an affected network without assessing the risk. Do not assume that paying a ransom guarantees recovery. Isolate affected systems safely and follow the incident response process."
        ),
        "hi": (
            "Ransomware से उबरने की तैयारी",
            "Ransomware से बचाव में शुरुआती access रोकना, privileges सीमित करना, असामान्य गतिविधि पहचानना और साफ systems restore करना शामिल है। Offline या immutable backups रखें और restoration test करें। घटना की आशंका पर approved response plan अपनाएँ और बदलाव से पहले evidence सुरक्षित रखें।",
            "Test किए हुए backups रखें, backup credentials अलग रखें, exposed systems patch करें, remote access सीमित करें, बड़े पैमाने पर file changes monitor करें और service owners के साथ recovery rehearsal करें।",
            "जोखिम जाँचे बिना backup storage को प्रभावित network से न जोड़ें। Ransom देने से recovery की गारंटी नहीं मिलती। प्रभावित systems को सुरक्षित तरीके से isolate करें और incident response प्रक्रिया अपनाएँ।"
        )
    },
    {
        "en": (
            "incident response triage",
            "Incident triage establishes what happened, which assets may be affected and how urgent the situation is. Use verified evidence rather than assumptions. Record timestamps, affected accounts, system identifiers and actions taken so the response team can coordinate containment and recovery.",
            "Confirm the alert source, preserve relevant logs, identify affected assets and accounts, assign an incident owner, assess business impact, and record every containment action.",
            "Do not wipe or rebuild a potentially affected system before evidence needs are considered. Coordinate with the incident lead, preserve logs and follow the organization's evidence-handling rules."
        ),
        "hi": (
            "Incident response की शुरुआती जाँच",
            "Incident triage यह समझने में मदद करती है कि क्या हुआ, कौन-से assets प्रभावित हो सकते हैं और स्थिति कितनी गंभीर है। अनुमान के बजाय सत्यापित evidence उपयोग करें। Timestamps, प्रभावित accounts, system identifiers और की गई कार्रवाइयाँ दर्ज करें ताकि team मिलकर containment और recovery कर सके।",
            "Alert का source confirm करें, संबंधित logs सुरक्षित रखें, प्रभावित assets और accounts पहचानें, incident owner तय करें, business impact जाँचें और हर containment action दर्ज करें।",
            "Evidence की आवश्यकता समझे बिना संभावित रूप से प्रभावित system को wipe या rebuild न करें। Incident lead से समन्वय करें, logs सुरक्षित रखें और संगठन के evidence-handling नियम अपनाएँ।"
        )
    },
    {
        "en": (
            "security logging",
            "Useful security logs help explain who did what, when and from which system. Centralize important events, synchronize clocks, restrict log access and protect records against tampering. Collect only data needed for security and operations, and define a retention period that matches policy.",
            "Log authentication, privilege changes and critical configuration changes; synchronize time sources; restrict write access; alert on high-risk events; and test that logs can be retrieved.",
            "Do not collect passwords, session secrets or unnecessary personal data in logs. Mask sensitive fields, restrict access, and check that logging failures are visible to operators."
        ),
        "hi": (
            "Security logging",
            "उपयोगी security logs बताते हैं कि किसने क्या किया, कब किया और किस system से किया। महत्वपूर्ण events को centralize करें, clocks synchronize रखें, log access सीमित करें और records को छेड़छाड़ से सुरक्षित रखें। केवल आवश्यक data collect करें और policy के अनुसार retention अवधि तय करें।",
            "Authentication, privilege changes और महत्वपूर्ण configuration changes log करें; time sources synchronize करें; write access सीमित रखें; high-risk events पर alert बनाएँ और logs की retrieval जाँचें।",
            "Logs में passwords, session secrets या अनावश्यक personal data न रखें। Sensitive fields mask करें, access सीमित करें और logging failure operators को दिखाई दे यह सुनिश्चित करें।"
        )
    },
    {
        "en": (
            "access reviews",
            "An access review checks whether each account still needs its current permissions. Reviews are especially important after role changes, project completion, contractor departure or organizational restructuring. Use an accountable owner and record the decision to retain, reduce or remove access.",
            "Compare accounts with current role and team membership, review privileged access first, confirm service-account owners, remove stale permissions, and record reviewer approval.",
            "Do not approve every permission simply because it was present in the previous review. Ask whether the current job still requires it and remove access that has no clear owner or purpose."
        ),
        "hi": (
            "Access review",
            "Access review में जाँचा जाता है कि हर account को अभी भी दी गई permissions की आवश्यकता है या नहीं। भूमिका बदलने, project पूरा होने, contractor के जाने या संगठन में बदलाव के बाद review विशेष रूप से जरूरी है। निर्णय के लिए जिम्मेदार owner रखें और access बनाए रखने, घटाने या हटाने का कारण दर्ज करें।",
            "Accounts की तुलना वर्तमान भूमिका और team membership से करें, privileged access पहले जाँचें, service-account owners की पुष्टि करें, पुरानी permissions हटाएँ और reviewer approval दर्ज करें।",
            "पिछले review में permission थी इसलिए उसे फिर से approve न करें। जाँचें कि वर्तमान काम के लिए उसकी जरूरत है या नहीं और बिना स्पष्ट owner या उद्देश्य वाला access हटा दें।"
        )
    },
    {
        "en": (
            "data classification",
            "Data classification groups information by sensitivity and business impact. It helps determine who may access data, where it may be stored, how it should be shared and when it must be deleted. Use clear labels and handling rules that employees can apply consistently.",
            "Define a small set of sensitivity levels, assign data owners, label important repositories, map each level to storage and sharing rules, and review classifications after major changes.",
            "Do not label all information as highly confidential by default. Over-classification makes controls hard to follow. Give examples for each level and ask owners to review unclear cases."
        ),
        "hi": (
            "Data classification",
            "Data classification जानकारी को उसकी संवेदनशीलता और business impact के अनुसार समूहों में बाँटती है। इससे तय होता है कि कौन data देख सकता है, वह कहाँ store होगा, कैसे share होगा और कब delete करना है। स्पष्ट labels और ऐसे नियम बनाएँ जिन्हें कर्मचारी एक समान तरीके से अपना सकें।",
            "संवेदनशीलता के कुछ स्पष्ट स्तर तय करें, data owners नियुक्त करें, महत्वपूर्ण repositories label करें, हर स्तर के storage और sharing नियम लिखें और बड़े बदलाव के बाद classification review करें।",
            "हर जानकारी को default रूप से अत्यधिक गोपनीय न मानें। जरूरत से ज्यादा classification नियमों को कठिन बनाती है। हर स्तर के उदाहरण दें और अस्पष्ट मामलों की owner से समीक्षा कराएँ।"
        )
    },
    {
        "en": (
            "secure API design",
            "A secure API authenticates callers, authorizes each requested action and validates input at the server boundary. Apply rate limits, return only necessary data and avoid exposing internal error details. Treat every client as untrusted, even when the user interface already checks permissions.",
            "Use short-lived credentials, check object-level authorization, validate schemas and sizes, set rate limits, redact secrets from logs, and test access-denial paths.",
            "Do not rely on a hidden button or client-side check as authorization. A caller can invoke an API directly. Enforce permissions on the server for every object and operation."
        ),
        "hi": (
            "सुरक्षित API design",
            "सुरक्षित API caller की authentication करती है, हर requested action की authorization जाँचती है और server boundary पर input validate करती है। Rate limits रखें, केवल आवश्यक data लौटाएँ और internal error details उजागर न करें। हर client को untrusted मानें, भले ही user interface permission जाँचती हो।",
            "Short-lived credentials रखें, object-level authorization जाँचें, schemas और sizes validate करें, rate limits लगाएँ, logs से secrets हटाएँ और access-denial paths test करें।",
            "छिपे हुए button या client-side check को authorization न मानें। Caller API को सीधे invoke कर सकता है। हर object और operation पर server-side permission लागू करें।"
        )
    },
    {
        "en": (
            "cross-site scripting prevention",
            "Cross-site scripting can occur when an application places untrusted content into a page as executable markup or script. Use context-aware output encoding and safe templating. A content security policy is an additional layer, not a replacement for correct handling of untrusted data.",
            "Encode output for its HTML, attribute, URL or script context, avoid unsafe DOM APIs, sanitize rich text with a maintained library, and test pages with untrusted input.",
            "Do not assume input validation alone prevents cross-site scripting. Data can become unsafe in a different output context. Encode at the point of rendering and avoid executing user-controlled strings."
        ),
        "hi": (
            "Cross-site scripting से बचाव",
            "Cross-site scripting तब हो सकता है जब application अविश्वसनीय content को page में executable markup या script की तरह रख दे। Context-aware output encoding और सुरक्षित templates उपयोग करें। Content security policy अतिरिक्त परत है, untrusted data की सही handling का विकल्प नहीं।",
            "HTML, attribute, URL या script context के अनुसार output encode करें, unsafe DOM APIs से बचें, rich text को maintained library से sanitize करें और untrusted input के साथ pages test करें।",
            "केवल input validation को XSS का पूरा बचाव न मानें। अलग output context में data असुरक्षित हो सकता है। Rendering के समय encode करें और user-controlled strings execute न करें।"
        )
    },
    {
        "en": (
            "secure wireless networks",
            "A secure wireless network uses modern encryption, a protected administrator interface and a separate guest network. Change default router credentials, keep firmware current and disable features that are not required. Guest devices should not have unrestricted access to internal business systems.",
            "Use WPA3 where supported, set a unique strong Wi-Fi password, change router admin credentials, update firmware, disable unused remote management, and isolate guest access.",
            "Do not share the administrator password as the Wi-Fi password. Anyone who knows it could change network settings. Keep management access separate and provide guests with a restricted network."
        ),
        "hi": (
            "सुरक्षित Wi-Fi network",
            "सुरक्षित wireless network में आधुनिक encryption, सुरक्षित administrator interface और अलग guest network होना चाहिए। Router के default credentials बदलें, firmware update रखें और अनावश्यक features बंद करें। Guest devices को internal business systems तक खुला access न दें।",
            "जहाँ उपलब्ध हो WPA3 उपयोग करें, अलग मजबूत Wi-Fi password रखें, router admin credentials बदलें, firmware update करें, अनावश्यक remote management बंद करें और guest access अलग रखें।",
            "Administrator password को Wi-Fi password की तरह साझा न करें। उसे जानने वाला network settings बदल सकता है। Management access अलग रखें और guests को सीमित network दें।"
        )
    },
    {
        "en": (
            "encryption at rest",
            "Encryption at rest protects stored data if storage media or a device is lost or accessed outside its normal environment. It works best with controlled key access, strong authentication and clear recovery procedures. Encryption does not prevent an authorized but compromised application from reading data.",
            "Identify sensitive storage, enable supported disk or database encryption, protect keys separately, restrict key administrators, test recovery, and revoke keys when devices are retired.",
            "Do not store the only decryption key beside the encrypted backup. Separate key storage from protected data, restrict who can use it, and document how authorized recovery works."
        ),
        "hi": (
            "Stored data का encryption",
            "Encryption at rest stored data की रक्षा करता है यदि storage media या device खो जाए या सामान्य वातावरण से बाहर access हो। यह तभी प्रभावी है जब key access नियंत्रित हो, authentication मजबूत हो और recovery प्रक्रिया स्पष्ट हो। Encryption compromised लेकिन authorized application को data पढ़ने से अपने-आप नहीं रोकता।",
            "संवेदनशील storage पहचानें, disk या database encryption चालू करें, keys अलग सुरक्षित रखें, key administrators सीमित करें, recovery test करें और retired devices की keys revoke करें।",
            "Encrypted backup की अकेली decryption key उसी backup के पास न रखें। Key storage को protected data से अलग रखें, उपयोग करने वालों को सीमित करें और authorized recovery लिखित रखें।"
        )
    },
    {
        "en": (
            "zero-trust security",
            "Zero-trust security verifies each access request using identity, device condition, context and policy instead of assuming that a network location is trustworthy. It is a continuing architecture, not a single product. Start with critical resources and improve visibility before adding complex rules.",
            "Inventory critical applications, require strong identity checks, assess device health, apply least privilege, log access decisions, and review policies using real usage data.",
            "Do not simply block every request and call it zero trust. Rules should be based on identity and resource risk, tested with service owners and adjusted to avoid unsafe workarounds."
        ),
        "hi": (
            "Zero-trust security",
            "Zero-trust में हर access request को identity, device की स्थिति, context और policy के आधार पर verify किया जाता है। केवल network location देखकर भरोसा नहीं किया जाता। यह लगातार विकसित होने वाली architecture है, कोई एक product नहीं। Critical resources से शुरुआत करें और जटिल नियम जोड़ने से पहले visibility बढ़ाएँ।",
            "महत्वपूर्ण applications की inventory बनाएँ, मजबूत identity checks लागू करें, device health जाँचें, least privilege अपनाएँ, access decisions log करें और वास्तविक usage के आधार पर policies review करें।",
            "हर request block करके उसे zero trust न कहें। नियम identity और resource के जोखिम पर आधारित हों, service owners के साथ test हों और असुरक्षित workarounds से बचने के लिए सुधारे जाएँ।"
        )
    },
    {
        "en": (
            "secure software development",
            "Secure software development builds security into design, coding, review, testing and release. Use threat modeling to identify important assets and trust boundaries. Protect build credentials, review dependencies and keep a clear process for reporting and fixing vulnerabilities.",
            "Define security requirements, review high-risk design changes, scan dependencies, require peer review, protect build pipelines, and record release approvals.",
            "Do not treat a clean automated scan as proof that software is secure. Scanners miss design flaws and business-logic issues. Combine automated checks with code review, testing and threat analysis."
        ),
        "hi": (
            "सुरक्षित software development",
            "Secure software development में design, coding, review, testing और release के हर चरण में security शामिल रहती है। Threat modeling से महत्वपूर्ण assets और trust boundaries पहचानें। Build credentials सुरक्षित रखें, dependencies की समीक्षा करें और vulnerabilities report तथा fix करने की स्पष्ट प्रक्रिया रखें।",
            "Security requirements तय करें, high-risk design changes review करें, dependencies scan करें, peer review अनिवार्य करें, build pipelines सुरक्षित रखें और release approvals दर्ज करें।",
            "Automated scan साफ आने को software की पूरी सुरक्षा का प्रमाण न मानें। Scanner design flaws और business-logic समस्याएँ छोड़ सकता है। Automated checks के साथ code review, testing और threat analysis भी करें।"
        )
    },
    {
        "en": (
            "cloud storage permissions",
            "Cloud storage should be private by default and grant access only to named users, services or approved groups. Public sharing should be a deliberate exception with an owner and expiry. Monitor permission changes and review links that allow access without authentication.",
            "Disable public access by default, restrict sharing domains, use short-lived links, log permission changes, scan for exposed buckets, and review access after projects end.",
            "Do not use an unlisted or hard-to-guess URL as the only protection for confidential files. Links can be copied or leaked. Require authentication and explicit authorization for sensitive content."
        ),
        "hi": (
            "Cloud storage permissions",
            "Cloud storage default रूप से private होनी चाहिए और access केवल तय users, services या approved groups को मिले। Public sharing स्पष्ट अपवाद हो, जिसका owner और expiry हो। Permission changes monitor करें और बिना authentication वाले links की समीक्षा करें।",
            "Default public access बंद रखें, sharing domains सीमित करें, short-lived links उपयोग करें, permission changes log करें, खुले buckets scan करें और project खत्म होने पर access review करें।",
            "Confidential files की सुरक्षा के लिए केवल छिपे या कठिन URL पर भरोसा न करें। Link copy या leak हो सकता है। Sensitive content के लिए authentication और स्पष्ट authorization अनिवार्य करें।"
        )
    },
    {
        "en": (
            "identity lifecycle management",
            "Identity lifecycle management creates, changes and disables accounts as people join, change roles or leave. Timely offboarding prevents former users and unused service accounts from retaining access. Coordinate the process across managers, human resources, IT and application owners.",
            "Use an approved joiner-mover-leaver process, assign an account owner, disable access promptly at departure, transfer business data, revoke tokens, and confirm application-specific accounts are closed.",
            "Do not disable only the main directory account and assume all access has ended. Check cloud applications, API tokens, mobile devices, certificates and third-party accounts as part of offboarding."
        ),
        "hi": (
            "Identity lifecycle management",
            "Identity lifecycle management में व्यक्ति के जुड़ने, भूमिका बदलने या संगठन छोड़ने पर accounts बनाए, बदले और बंद किए जाते हैं। समय पर offboarding से पूर्व कर्मचारियों और unused service accounts का access समाप्त होता है। Managers, HR, IT और application owners मिलकर प्रक्रिया चलाएँ।",
            "Approved joiner-mover-leaver प्रक्रिया अपनाएँ, account owner तय करें, departure पर access तुरंत disable करें, business data transfer करें, tokens revoke करें और application-specific accounts बंद होने की पुष्टि करें।",
            "केवल मुख्य directory account बंद करके यह न मानें कि सारा access समाप्त हो गया। Offboarding में cloud apps, API tokens, mobile devices, certificates और third-party accounts भी जाँचें।"
        )
    },
    {
        "en": (
            "backup restoration testing",
            "A backup is useful only if the organization can restore the required data within an acceptable time. Test complete restoration, not just whether a backup job created a file. Include application configuration, encryption keys, dependencies and business-owner validation.",
            "Define recovery objectives, select representative systems, restore into an isolated environment, verify data and application function, measure recovery time, and record lessons.",
            "Do not wait for an outage to discover that a backup is corrupted or that its key is missing. Schedule regular restore tests and assign an owner to track every corrective action."
        ),
        "hi": (
            "Backup restoration test",
            "Backup तभी उपयोगी है जब संगठन आवश्यक data को स्वीकार्य समय में restore कर सके। केवल backup job की file बन जाने की जाँच पर्याप्त नहीं है। Complete restoration test करें, जिसमें application configuration, encryption keys, dependencies और business-owner validation शामिल हों।",
            "Recovery objectives तय करें, representative systems चुनें, isolated environment में restore करें, data और application function verify करें, recovery time मापें और सीख दर्ज करें।",
            "Outage के समय यह पता चलने की प्रतीक्षा न करें कि backup corrupt है या key गायब है। नियमित restore tests schedule करें और हर सुधारात्मक कार्रवाई का owner तय करें।"
        )
    },
    {
        "en": (
            "vulnerability disclosure",
            "A useful vulnerability report explains the affected product and version, the security impact, safe reproduction conditions and supporting evidence. Share it privately with the designated security contact. Avoid accessing other people's data, disrupting service or publishing exploit details before a coordinated fix.",
            "Record the affected component, version and environment, describe impact, include minimal redacted evidence, use the official reporting channel, and agree on a communication timeline.",
            "Do not include passwords, personal records or unrelated customer data in a report. Redact sensitive evidence and stop testing once impact is established. Follow the owner's coordinated disclosure process."
        ),
        "hi": (
            "Vulnerability disclosure",
            "उपयोगी vulnerability report में प्रभावित product और version, security impact, सुरक्षित reproduction conditions और supporting evidence शामिल होते हैं। इसे तय security contact को private रूप से भेजें। दूसरे लोगों का data access न करें, service बाधित न करें और coordinated fix से पहले exploit details सार्वजनिक न करें।",
            "Affected component, version और environment दर्ज करें, impact समझाएँ, न्यूनतम redacted evidence दें, official reporting channel अपनाएँ और communication timeline पर सहमति बनाएँ।",
            "Report में passwords, personal records या दूसरे customers का data न जोड़ें। Sensitive evidence redact करें और impact स्थापित होते ही testing रोकें। Owner की coordinated disclosure प्रक्रिया अपनाएँ।"
        )
    },
    {
        "en": (
            "endpoint security",
            "Endpoint security protects laptops, servers and mobile devices through configuration, patching, malware detection and device controls. A managed endpoint should report its health and receive updates. Use encryption and screen locks, and isolate a device when the response team confirms it is necessary.",
            "Enroll devices in management, enable disk encryption, apply security updates, use endpoint detection, restrict local administrator rights, and monitor devices that stop reporting.",
            "Do not install random security tools on a managed device. They can conflict with approved controls or expose data. Contact IT and use the organization's approved software catalogue."
        ),
        "hi": (
            "Endpoint security",
            "Endpoint security laptops, servers और mobile devices की सुरक्षा configuration, patching, malware detection और device controls से करती है। Managed endpoint को अपनी health report करनी चाहिए और updates मिलने चाहिए। Encryption तथा screen lock रखें और response team की पुष्टि पर device isolate करें।",
            "Devices को management में enroll करें, disk encryption चालू रखें, security updates लगाएँ, endpoint detection उपयोग करें, local administrator rights सीमित करें और report न करने वाले devices monitor करें।",
            "Managed device पर अनजान security tools install न करें। वे approved controls से टकरा सकते हैं या data expose कर सकते हैं। IT से संपर्क करें और approved software catalogue उपयोग करें।"
        )
    },
    {
        "en": (
            "security awareness training",
            "Security awareness works best when it teaches people how to make safe decisions in realistic situations. Give clear reporting routes and praise early reporting, including mistaken clicks. Training should be short, accessible and repeated when tools or threats change.",
            "Use role-specific examples, explain how to report a concern, practise verification of unusual requests, provide accessible materials, and measure reporting quality rather than click rates alone.",
            "Do not shame employees who report a mistake. Fear discourages reporting and delays containment. Respond respectfully, gather facts and improve the process that allowed the mistake."
        ),
        "hi": (
            "Security awareness training",
            "Security awareness तब उपयोगी होती है जब वह वास्तविक परिस्थितियों में सुरक्षित निर्णय लेना सिखाती है। स्पष्ट reporting रास्ता दें और गलती से click होने पर भी जल्दी report करने को प्रोत्साहित करें। Training छोटी, सुलभ और tools या threats बदलने पर दोहराई जानी चाहिए।",
            "भूमिका के अनुसार उदाहरण दें, concern report करने का तरीका समझाएँ, असामान्य अनुरोध सत्यापित करने का अभ्यास कराएँ, सुलभ सामग्री दें और केवल click rate नहीं बल्कि reporting quality भी मापें।",
            "गलती report करने वाले कर्मचारी को शर्मिंदा न करें। डर से reporting रुकती है और containment देर से होता है। सम्मान से तथ्य जुटाएँ और गलती संभव बनाने वाली प्रक्रिया सुधारें।"
        )
    },
    {
        "en": (
            "security monitoring alerts",
            "A security alert is a signal that needs context, not automatic proof of an attack. Analysts should check the source, affected identity, asset criticality and related events. Tune noisy detections carefully while keeping enough coverage to catch meaningful behaviour.",
            "Confirm alert time and source, correlate related events, check whether the activity was expected, record the decision, and escalate when impact or uncertainty is high.",
            "Do not close repeated alerts automatically because similar alerts were false positives before. Check whether the context changed and retain a sample for detection tuning."
        ),
        "hi": (
            "Security monitoring alerts",
            "Security alert एक संकेत है जिसकी context जाँचनी होती है; यह अपने-आप attack का प्रमाण नहीं है। Analyst source, प्रभावित identity, asset की महत्ता और संबंधित events देखें। Noise कम करने के लिए detection सावधानी से tune करें, लेकिन जरूरी व्यवहार पकड़ने की क्षमता बनाए रखें।",
            "Alert का समय और source confirm करें, संबंधित events जोड़कर देखें, activity अपेक्षित थी या नहीं जाँचें, निर्णय दर्ज करें और impact या uncertainty अधिक हो तो escalate करें।",
            "पहले false positive आया था इसलिए हर नए alert को स्वतः बंद न करें। जाँचें कि context बदली है या नहीं और detection सुधारने के लिए एक sample सुरक्षित रखें।"
        )
    },
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--heldout-out", type=Path, required=True)
    args = ap.parse_args()
    train_rows, heldout_rows = [], []
    split = len(TOPICS) - 6
    for topic_index, topic in enumerate(TOPICS):
        for lang in ("en", "hi"):
            label, explanation, checklist, pitfall = topic[lang]
            if lang == "en":
                variants = [
                    (f"Explain {label} in plain language.", explanation),
                    (f"Give a practical checklist for {label}.", checklist),
                    (f"What common mistake should be avoided with {label}?", pitfall),
                ]
                prefix, answer_prefix = "User: ", "\\nAssistant: "
            else:
                variants = [
                    (f"{label} को सरल भाषा में समझाइए।", explanation),
                    (f"{label} के लिए एक व्यावहारिक checklist दीजिए।", checklist),
                    (f"{label} में कौन-सी सामान्य गलती से बचना चाहिए?", pitfall),
                ]
                prefix, answer_prefix = "उपयोगकर्ता: ", "\\nसहायक: "
            target = heldout_rows if topic_index >= split else train_rows
            for question, answer in variants:
                target.append({
                    "text": prefix + question + answer_prefix + answer,
                    "language": lang,
                    "source": "anviksha-curated-defensive-instructions-v1",
                    "topic": label,
                })

    def write_rows(path: Path, items):
        path.parent.mkdir(parents=True, exist_ok=True)
        seen = set()
        with path.open("w", encoding="utf-8") as handle:
            for row in items:
                digest = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
                if digest not in seen:
                    seen.add(digest)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\\n")
        return seen

    train_seen = write_rows(args.out, train_rows)
    heldout_seen = write_rows(args.heldout_out, heldout_rows)
    counts = {lang: sum(r["language"] == lang for r in train_rows) for lang in ("en", "hi")}
    heldout_counts = {lang: sum(r["language"] == lang for r in heldout_rows) for lang in ("en", "hi")}
    manifest = {
        "name": "ANVIKSHA bilingual defensive instruction replay v1",
        "training_topics": split,
        "heldout_topics": len(TOPICS) - split,
        "training_documents": counts,
        "heldout_documents": heldout_counts,
        "unique_training_documents": len(train_seen),
        "unique_heldout_documents": len(heldout_seen),
        "train_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "heldout_sha256": hashlib.sha256(args.heldout_out.read_bytes()).hexdigest(),
        "training_source": "human-authored original examples",
        "heldout_used_for_training": False,
    }
    args.out.with_suffix(args.out.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if min(counts.values()) < 50 or min(heldout_counts.values()) < 10:
        raise SystemExit("bilingual train/held-out corpus below minimum size")


if __name__ == "__main__":
    main()
