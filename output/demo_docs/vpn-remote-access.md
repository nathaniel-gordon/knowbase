# VPN and Remote Access Guide
Remote access to internal systems requires the GlobalConnect VPN client. Install it from the
self-service portal, sign in with SSO, and approve the MFA push. Split tunneling is disabled by
policy; all traffic routes through the corporate gateway. If the VPN disconnects repeatedly,
first switch from UDP to TCP mode in Settings, then verify your home router's MTU is at least
1400, and finally reinstall the client. Contractors receive time-boxed VPN certificates valid
for 30 days, renewable by their sponsoring manager. Access to the production subnet additionally
requires membership in the prod-access group, granted through a Tier-2 request with director
approval.
