# Task: fix(connectors): [authorizedotnet] send customerProfileId in the CIT flow and map customerPaymentProfileId in the Setup Mandate flow

## PR Information
**Title:** fix(connectors): [authorizedotnet] send customerProfileId in the CIT flow and map customerPaymentProfileId in the Setup Mandate flow

## Description
## Type of Change
<!-- Put an `x` in the boxes that apply -->

- [x] Bugfix
- [ ] New feature
- [ ] Enhancement
- [ ] Refactoring
- [ ] Dependency updates
- [ ] Documentation
- [ ] CI/CD

## Description
<!-- Describe your changes in detail -->
This PR fixes
1. Setup Mandate flow - Earlier Error is thrown in customer setup mandate if the payment profile is already present at the connector. Now, if `customerPaymentProfileId` is present we will map it to mandate id, if not we will return an error response. 

2. For CIT, pass `customerProfileId` and for normal payments pass the customer id and enable create profiles only for CIT

3. Handle the case of customer id not created from hyperswitch but used via hyperswitch

> Note: Ensure the customer id in the dashboard. Incase of CIT customer id will not be mapped to the transaction instead the payment profile will be stored under the customer

## How did you test it?
<!--
Did you write an integration/unit/API test to verify the code changes?
Or did you test this change manually (provide relevant screenshots)?
-->
<details>
<summary>Test Setup mandate </summary> 

1. Create a setup mandate payment for a customer

```
curl --location 'http://localhost:8080/payments' \
--header 'Content-Type: application/json' \
--header 'Accept: application/json' \
--header 'api-key: dev_QTV7hfwJvBw7V1Cw4IxHA8h1g**********' \
--data '{
    "amount": 0,
    "currency": "USD",
    "confirm": true,
    "description": "oda",
    
    "capture_method": "automatic",
    
    "payment_type": "setup_mandate",
    "customer_id": "aaa9",
    "payment_method": "card",
    "payment_method_type": "credit",
 
    
    "payment_method_data": {
        "card": {
            "card_number": "4000000000001091",
            "card_exp_month": "01",
            "card_exp_year": "30",
            
            "card_cvc": "123"
            
        }
    },
    
    "customer_acceptance": {
        "acceptance_type": "online",
        "accepted_at": "2025-03-27T13:56:49.848Z",
        "online": {
            "ip_address": null,
            "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.3 Safari/605.1.15"
        }
    },
    "setup_future_usage": "off_session",
    

    "browser_info": {
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36",
        "accept_header": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "language": "nl-NL",
        "color_depth": 24,
        "screen_height": 723,
        "screen_width": 1536,
        "time_zone": 0,
        "java_enabled": true,
        "java_script_enabled": true,
        "ip_address": "13.232.74.226"
    }    
}'
```
Response 

```
{
    "payment_id": "pay_2ZOEDHarHH50bJ0CsShD",
    "merchant_id": "postman_merchant_GHAction_7ca00208-f80b-4f92-8ba9-3cbf9b0eb6ed",
    "status": "succeeded",
    "amount": 0,
    "net_amount": 0,
    "shipping_cost": null,
    "amount_capturable": 0,
    "amount_received": null,
    "connector": "authorizedotnet",
    "client_secret": "pay_2ZOEDHarHH50bJ0CsShD_secret_PvsjaSnomlWtvyxCBu4N",
    "created": "2025-09-08T06:29:33.850Z",
    "currency": "USD",
    "customer_id": "aaa9",
    "customer": {
        "id": "aaa9",
        "name": null,
        "email": null,
        "phone": null,
        "phone_country_code": null
    },
    "description": "oda",
    "refunds": null,
    "disputes": null,
    "mandate_id": null,
    "mandate_data": null,
    "setup_future_usage": "off_session",
    "off_session": null,
    "capture_on": null,
    "capture_method": "automatic",
    "payment_method": "card",
    "payment_method_data": {
        "card": {
            "last4": "1091",
            "card_type": null,
            "card_network": null,
            "card_issuer": null,
            "card_issuing_country": null,
            "card_isin": "400000",
            "card_extended_bin": null,
            "card_exp_month": "01",
            "card_exp_year": "30",
            "card_holder_name": null,
            "payment_checks": null,
            "authentication_data": null
        },
        "billing": null
    },
    "payment_token": null,
    "shipping": null,
    "billing": null,
    "order_details": null,
    "email": null,
    "name": null,
    "phone": null,
    "return_url": null,
    "authentication_type": "no_three_ds",
    "statement_descriptor_name": null,
    "statement_descriptor_suffix": null,
    "next_action": null,
    "cancellation_reason": null,
    "error_code": null,
    "error_message": null,
    "unified_code": null,
    "unified_message": null,
    "payment_experience": null,
    "payment_method_type": "credit",
    "connector_label": null,
    "business_country": null,
    "business_label": "default",
    "business_sub_label": null,
    "allowed_payment_method_types": null,
    "ephemeral_key": {
        "customer_id": "aaa9",
        "created_at": 1757312973,
        "expires": 1757316573,
        "secret": "epk_348e3583ce864cea9a6d4d4c882464db"
    },
    "manual_retry_allowed": false,
    "connector_transaction_id": null,
    "frm_message": null,
    "metadata": null,
    "connector_metadata": null,
    "feature_metadata": {
        "redirect_response": null,
        "search_tags": null,
        "apple_pay_recurring_details": null,
        "gateway_system": "direct"
    },
    "reference_id": null,
    "payment_link": null,
    "profile_id": "pro_B1KJWgaKMyj9NJZOiGiL",
    "surcharge_details": null,
    "attempt_count": 1,
    "merchant_decision": null,
    "merchant_connector_id": "mca_1A9mFebPqoV963nNNKk4",
    "incremental_authorization_allowed": false,
    "authorization_count": null,
    "incremental_authorizations": null,
    "external_authentication_details": null,
    "external_3ds_authentication_attempted": false,
    "expires_on": "2025-09-08T06:44:33.849Z",
    "fingerprint": null,
    "browser_info": {
        "language": "nl-NL",
        "time_zone": 0,
        "ip_address": "13.232.74.226",
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36",
        "color_depth": 24,
        "java_enabled": true,
        "screen_width": 1536,
        "accept_header": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "screen_height": 723,
        "java_script_enabled": true
    },
    "payment_channel": null,
    "payment_method_id": "pm_DFYzCRBsmfEIMZ40aHuk",
    "network_transaction_id": null,
    "payment_method_status": "active",
    "updated": "2025-09-08T06:29:35.954Z",
    "split_payments": null,
    "frm_metadata": null,
    "extended_authorization_applied": null,
    "capture_before": null,
    "merchant_order_reference_id": null,
    "order_tax_amount": null,
    "connector_mandate_id": "932578546-931871295",
    "card_discovery": "manual",
    "force_3ds_challenge": false,
    "force_3ds_challenge_trigger": false,
    "issuer_error_code": null,
    "issuer_error_message": null,
    "is_iframe_redirection_enabled": null,
    "whole_connector_response": null,
    "enable_partial_authorization": null
}
```

Now Create another setup mandate payment for same customer

```
curl --location 'http://localhost:8080/payments' \
--header 'Content-Type: application/json' \
--header 'Accept: application/json' \
--header 'api-key: dev_QTV7hfwJvBw7V1Cw4IxHA8h1gg9in8dKYTNWtoWa6BkgfjvSr5r8Jg9gnZWL0wzX' \
--data '{
    "amount": 0,
    "currency": "USD",
    "confirm": true,
    "description": "oda",
    
    "capture_method": "automatic",
    
    "payment_type": "setup_mandate",
    "customer_id": "aaa9",
    "payment_method": "card",
    "payment_method_type": "credit",
    "payment_method_data": {
        "card": {
            "card_number": "4000000000001091",
            "card_exp_month": "01",
            "card_exp_year": "30",
            
            "card_cvc": "123"
            
        }
    },
    
    "customer_acceptance": {
        "acceptance_type": "online",
        "accepted_at": "2025-03-27T13:56:49.848Z",
        "online": {
            "ip_address": null,
            "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.3 Safari/605.1.15"
        }
    },
    "setup_future_usage": "off_session",
    "browser_info": {
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36",
        "accept_header": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "language": "nl-NL",
        "color_depth": 24,
        "screen_height": 723,
        "screen_width": 1536,
        "time_zone": 0,
        "java_enabled": true,
        "java_script_enabled": true,
        "ip_address": "13.232.74.226"
    }
}'
```
Response

```
{
    "payment_id": "pay_8S1upQ4fMllhyoljKbsy",
    "merchant_id": "postman_merchant_GHAction_7ca00208-f80b-4f92-8ba9-3cbf9b0eb6ed",
    "status": "succeeded",
    "amount": 0,
    "net_amount": 0,
    "shipping_cost": null,
    "amount_capturable": 0,
    "amount_received": null,
    "connector": "authorizedotnet",
    "client_secret": "pay_8S1upQ4fMllhyoljKbsy_secret_nMnZtHqYqXETjQR8Y5Wl",
    "created": "2025-09-08T06:35:42.276Z",
    "currency": "USD",
    "customer_id": "aaa9",
    "customer": {
        "id": "aaa9",
        "name": null,
        "email": null,
        "phone": null,
        "phone_country_code": null
    },
    "description": "oda",
    "refunds": null,
    "disputes": null,
    "mandate_id": null,
    "mandate_data": null,
    "setup_future_usage": "off_session",
    "off_session": null,
    "capture_on": null,
    "capture_method": "automatic",
    "payment_method": "card",
    "payment_method_data": {
        "card": {
            "last4": "1091",
            "card_type": null,
            "card_network": null,
            "card_issuer": null,
            "card_issuing_country": null,
            "card_isin": "400000",
            "card_extended_bin": null,
            "card_exp_month": "01",
            "card_exp_year": "30",
            "card_holder_name": null,
            "payment_checks": null,
            "authentication_data": null
        },
        "billing": null
    },
    "payment_token": null,
    "shipping": null,
    "billing": null,
    "order_details": null,
    "email": null,
    "name": null,
    "phone": null,
    "return_url": null,
    "authentication_type": "no_three_ds",
    "statement_descriptor_name": null,
    "statement_descriptor_suffix": null,
    "next_action": null,
    "cancellation_reason": null,
    "error_code": null,
    "error_message": null,
    "unified_code": null,
    "unified_message": null,
    "payment_experience": null,
    "payment_method_type": "credit",
    "connector_label": null,
    "business_country": null,
    "business_label": "default",
    "business_sub_label": null,
    "allowed_payment_method_types": null,
    "ephemeral_key": {
        "customer_id": "aaa9",
        "created_at": 1757313342,
        "expires": 1757316942,
        "secret": "epk_5ff5134eb9d145698421cd65302b777f"
    },
    "manual_retry_allowed": false,
    "connector_transaction_id": null,
    "frm_message": null,
    "metadata": null,
    "connector_metadata": null,
    "feature_metadata": {
        "redirect_response": null,
        "search_tags": null,
        "apple_pay_recurring_details": null,
        "gateway_system": "direct"
    },
    "reference_id": null,
    "payment_link": null,
    "profile_id": "pro_B1KJWgaKMyj9NJZOiGiL",
    "surcharge_details": null,
    "attempt_count": 1,
    "merchant_decision": null,
    "merchant_connector_id": "mca_1A9mFebPqoV963nNNKk4",
    "incremental_authorization_allowed": false,
    "authorization_count": null,
    "incremental_authorizations": null,
    "external_authentication_details": null,
    "external_3ds_authentication_attempted": false,
    "expires_on": "2025-09-08T06:50:42.276Z",
    "fingerprint": null,
    "browser_info": {
        "language": "nl-NL",
        "time_zone": 0,
        "ip_address": "13.232.74.226",
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36",
        "color_depth": 24,
        "java_enabled": true,
        "screen_width": 1536,
        "accept_header": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "screen_height": 723,
        "java_script_enabled": true
    },
    "payment_channel": null,
    "payment_method_id": "pm_dNGfqp6eJ3EQIXDKT8Ul",
    "network_transaction_id": null,
    "payment_method_status": "active",
    "updated": "2025-09-08T06:35:43.810Z",
    "split_payments": null,
    "frm_metadata": null,
    "extended_authorization_applied": null,
    "capture_before": null,
    "merchant_order_reference_id": null,
    "order_tax_amount": null,
    "connector_mandate_id": "932578546-931871295",
    "card_discovery": "manual",
    "force_3ds_challenge": false,
    "force_3ds_challenge_trigger": false,
    "issuer_error_code": null,
    "issuer_error_message": null,
    "is_iframe_redirection_enabled": null,
    "whole_connector_response": null,
    "enable_partial_authorization": null
}
```

3. MIT via payment method id
```
curl --location 'http://localhost:8080/payments' \
--header 'Content-Type: application/json' \
--header 'Accept: application/json' \
--header 'api-key: dev_QTV7hfwJvBw7V1Cw4IxHA8h1gg9in8dKYTNWtoWa6BkgfjvSr5r8Jg9gnZWL0wzX' \
--data '{
    "amount": 10000,
    "currency": "USD",
    "off_session": true,
    "confirm": true,
    "capture_method": "automatic",
    "recurring_details": {
        "type": "payment_method_id",
        "data": "pm_dNGfqp6eJ3EQIXDKT8Ul"
    },
    
    "customer_id": "aaa9",
    "connector": [
        "authorizedotnet"
    ]
    
}'
```
Response
```
{
    "payment_id": "pay_Sg03kzM6XooUiUlc76PE",
    "merchant_id": "postman_merchant_GHAction_7ca00208-f80b-4f92-8ba9-3cbf9b0eb6ed",
    "status": "succeeded",
    "amount": 10000,
    "net_amount": 10000,
    "shipping_cost": null,
    "amount_capturable": 0,
    "amount_received": 10000,
    "connector": "authorizedotnet",
    "client_secret": "pay_Sg03kzM6XooUiUlc76PE_secret_5cjMKfkDnpJFEd1FmSS8",
    "created": "2025-09-08T07:30:58.770Z",
    "currency": "USD",
    "customer_id": "aaa9",
    "customer": {
        "id": "aaa9",
        "name": null,
        "email": null,
        "phone": null,
        "phone_country_code": null
    },
    "description": null,
    "refunds": null,
    "disputes": null,
    "mandate_id": null,
    "mandate_data": null,
    "setup_future_usage": null,
    "off_session": true,
    "capture_on": null,
    "capture_method": "automatic",
    "payment_method": "card",
    "payment_method_data": {
        "card": {
            "last4": "1091",
            "card_type": null,
            "card_network": null,
            "card_issuer": null,
            "card_issuing_country": null,
            "card_isin": "400000",
            "card_extended_bin": null,
            "card_exp_month": "01",
            "card_exp_year": "30",
            "card_holder_name": null,
            "payment_checks": null,
            "authentication_data": null
        },
        "billing": null
    },
    "payment_token": null,
    "shipping": null,
    "billing": null,
    "order_details": null,
    "email": null,
    "name": null,
    "phone": null,
    "return_url": null,
    "authentication_type": "no_three_ds",
    "statement_descriptor_name": null,
    "statement_descriptor_suffix": null,
    "next_action": null,
    "cancellation_reason": null,
    "error_code": null,
    "error_message": null,
    "unified_code": null,
    "unified_message": null,
    "payment_experience": null,
    "payment_method_type": "credit",
    "connector_label": null,
    "business_country": null,
    "business_label": "default",
    "business_sub_label": null,
    "allowed_payment_method_types": null,
    "ephemeral_key": {
        "customer_id": "aaa9",
        "created_at": 1757316658,
        "expires": 1757320258,
        "secret": "epk_2fb224c3d65d4bfaaa95dc3f3a4f1829"
    },
    "manual_retry_allowed": false,
    "connector_transaction_id": "120070561048",
    "frm_message": null,
    "metadata": null,
    "connector_metadata": null,
    "feature_metadata": {
        "redirect_response": null,
        "search_tags": null,
        "apple_pay_recurring_details": null,
        "gateway_system": "direct"
    },
    "reference_id": "120070561048",
    "payment_link": null,
    "profile_id": "pro_B1KJWgaKMyj9NJZOiGiL",
    "surcharge_details": null,
    "attempt_count": 1,
    "merchant_decision": null,
    "merchant_connector_id": "mca_1A9mFebPqoV963nNNKk4",
    "incremental_authorization_allowed": false,
    "authorization_count": null,
    "incremental_authorizations": null,
    "external_authentication_details": null,
    "external_3ds_authentication_attempted": false,
    "expires_on": "2025-09-08T07:45:58.770Z",
    "fingerprint": null,
    "browser_info": null,
    "payment_channel": null,
    "payment_method_id": "pm_dNGfqp6eJ3EQIXDKT8Ul",
    "network_transaction_id": "IP5H1DF3H9BJM0ZYBF1JNZ6",
    "payment_method_status": "active",
    "updated": "2025-09-08T07:31:00.837Z",
    "split_payments": null,
    "frm_metadata": null,
    "extended_authorization_applied": null,
    "capture_before": null,
    "merchant_order_reference_id": null,
    "order_tax_amount": null,
    "connector_mandate_id": "932578546-931871295",
    "card_discovery": "manual",
    "force_3ds_challenge": false,
    "force_3ds_challenge_trigger": false,
    "issuer_error_code": null,
    "issuer_error_message": null,
    "is_iframe_redirection_enabled": null,
    "whole_connector_response": null,
    "enable_partial_authorization": null
}
```

</details>
<details>
<summary> Test CIT via same customer and payment method </summary>

```
curl --location 'http://localhost:8080/payments' \
--header 'Content-Type: application/json' \
--header 'Accept: application/json' \
--header 'api-key: dev_QTV7hfwJvBw7V1Cw4IxHA8h1gg9in8dKYTNWtoWa6BkgfjvSr5r8Jg9gnZWL0wzX' \
--data '{
    "amount": 1000,
    "currency": "USD",
    "confirm": true,
    "description": "oda",
    
    "capture_method": "automatic",

    "customer_id": "George_12",
    "payment_method": "card",
    "payment_method_type": "credit",
    
    "payment_method_data": {
        "card": {
            "card_number": "4000000000001091",
            "card_exp_month": "01",
            "card_exp_year": "30",
            
            "card_cvc": "123"
            
        }
    },
    
    "customer_acceptance": {
        "acceptance_type": "online",
        "accepted_at": "2025-03-27T13:56:49.848Z",
        "online": {
            "ip_address": null,
            "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.3 Safari/605.1.15"
        }
    },
    "setup_future_usage": "off_session",
    "browser_info": {
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36",
        "accept_header": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "language": "nl-NL",
        "color_depth": 24,
        "screen_height": 723,
        "screen_width": 1536,
        "time_zone": 0,
        "java_enabled": true,
        "java_script_enabled": true,
        "ip_address": "13.232.74.226"
    }
}'
```
Response
```
{
    "payment_id": "pay_cmywKrWMEyxjxor5GjGg",
    "merchant_id": "postman_merchant_GHAction_7ca00208-f80b-4f92-8ba9-3cbf9b0eb6ed",
    "status": "succeeded",
    "amount": 1000,
    "net_amount": 1000,
    "shipping_cost": null,
    "amount_capturable": 0,
    "amount_received": 1000,
    "connector": "authorizedotnet",
    "client_secret": "pay_cmywKrWMEyxjxor5GjGg_secret_p1LI3CglxmUMeDIfz31m",
    "created": "2025-09-08T07:33:47.317Z",
    "currency": "USD",
    "customer_id": "George_12",
    "customer": {
        "id": "George_12",
        "name": null,
        "email": null,
        "phone": null,
        "phone_country_code": null
    },
    "description": "oda",
    "refunds": null,
    "disputes": null,
    "mandate_id": null,
    "mandate_data": null,
    "setup_future_usage": "off_session",
    "off_session": null,
    "capture_on": null,
    "capture_method": "automatic",
    "payment_method": "card",
    "payment_method_data": {
        "card": {
            "last4": "1091",
            "card_type": null,
            "card_network": null,
            "card_issuer": null,
            "card_issuing_country": null,
            "card_isin": "400000",
            "card_extended_bin": null,
            "card_exp_month": "01",
            "card_exp_year": "30",
            "card_holder_name": null,
            "payment_checks": {
                "description": "The street address and postal code matched.",
                "avs_result_code": "Y"
            },
            "authentication_data": null
        },
        "billing": null
    },
    "payment_token": null,
    "shipping": null,
    "billing": null,
    "order_details": null,
    "email": null,
    "name": null,
    "phone": null,
    "return_url": null,
    "authentication_type": "no_three_ds",
    "statement_descriptor_name": null,
    "statement_descriptor_suffix": null,
    "next_action": null,
    "cancellation_reason": null,
    "error_code": null,
    "error_message": null,
    "unified_code": null,
    "unified_message": null,
    "payment_experience": null,
    "payment_method_type": "credit",
    "connector_label": null,
    "business_country": null,
    "business_label": "default",
    "business_sub_label": null,
    "allowed_payment_method_types": null,
    "ephemeral_key": {
        "customer_id": "George_12",
        "created_at": 1757316827,
        "expires": 1757320427,
        "secret": "epk_bd1723532d05441c90f833fe535e8ae8"
    },
    "manual_retry_allowed": false,
    "connector_transaction_id": "120070561131",
    "frm_message": null,
    "metadata": null,
    "connector_metadata": null,
    "feature_metadata": {
        "redirect_response": null,
        "search_tags": null,
        "apple_pay_recurring_details": null,
        "gateway_system": "direct"
    },
    "reference_id": "120070561131",
    "payment_link": null,
    "profile_id": "pro_B1KJWgaKMyj9NJZOiGiL",
    "surcharge_details": null,
    "attempt_count": 1,
    "merchant_decision": null,
    "merchant_connector_id": "mca_1A9mFebPqoV963nNNKk4",
    "incremental_authorization_allowed": false,
    "authorization_count": null,
    "incremental_authorizations": null,
    "external_authentication_details": null,
    "external_3ds_authentication_attempted": false,
    "expires_on": "2025-09-08T07:48:47.317Z",
    "fingerprint": null,
    "browser_info": {
        "language": "nl-NL",
        "time_zone": 0,
        "ip_address": "13.232.74.226",
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36",
        "color_depth": 24,
        "java_enabled": true,
        "screen_width": 1536,
        "accept_header": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "screen_height": 723,
        "java_script_enabled": true
    },
    "payment_channel": null,
    "payment_method_id": "pm_mfdEZEUD6TRHnl3xbXen",
    "network_transaction_id": "KKC9N0P1V9FZ8I4224LTENH",
    "payment_method_status": "active",
    "updated": "2025-09-08T07:33:48.962Z",
    "split_payments": null,
    "frm_metadata": null,
    "extended_authorization_applied": null,
    "capture_before": null,
    "merchant_order_reference_id": null,
    "order_tax_amount": null,
    "connector_mandate_id": "932579889-931872607",
    "card_discovery": "manual",
    "force_3ds_challenge": false,
    "force_3ds_challenge_trigger": false,
    "issuer_error_code": null,
    "issuer_error_message": null,
    "is_iframe_redirection_enabled": null,
    "whole_connector_response": null,
    "enable_partial_authorization": null
}
```

Create another CIT with authorize.net
```
curl --location 'http://localhost:8080/payments' \
--header 'Content-Type: application/json' \
--header 'Accept: application/json' \
--header 'api-key: dev_QTV7hfwJvBw7V1Cw4IxHA8h1gg9in8dKYTNWtoWa6BkgfjvSr5r8Jg9gnZWL0wzX' \
--data '{
    "amount": 1000,
    "currency": "USD",
    "confirm": true,
    "description": "oda",
    
    "capture_method": "automatic",

    
    "customer_id": "George_12",
    "payment_method": "card",
    "payment_method_type": "credit",
   
    
    "payment_method_data": {
        "card": {
            "card_number": "4000000000001091",
            "card_exp_month": "01",
            "card_exp_year": "30",
            
            "card_cvc": "123"
            
        }
    },
    
    "customer_acceptance": {
        "acceptance_type": "online",
        "accepted_at": "2025-03-27T13:56:49.848Z",
        "online": {
            "ip_address": null,
            "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.3 Safari/605.1.15"
        }
    },
    "setup_future_usage": "off_session",
    
    "browser_info": {
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36",
        "accept_header": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "language": "nl-NL",
        "color_depth": 24,
        "screen_height": 723,
        "screen_width": 1536,
        "time_zone": 0,
        "java_enabled": true,
        "java_script_enabled": true,
        "ip_address": "13.232.74.226"
    }

    
}'
```
Response
```
{
    "payment_id": "pay_CD6jJdF4ezCGlCtrpvA9",
    "merchant_id": "postman_merchant_GHAction_7ca00208-f80b-4f92-8ba9-3cbf9b0eb6ed",
    "status": "succeeded",
    "amount": 1000,
    "net_amount": 1000,
    "shipping_cost": null,
    "amount_capturable": 0,
    "amount_received": 1000,
    "connector": "authorizedotnet",
    "client_secret": "pay_CD6jJdF4ezCGlCtrpvA9_secret_3bQzVFajiF0grVvRqniB",
    "created": "2025-09-08T07:58:27.079Z",
    "currency": "USD",
    "customer_id": "George_12",
    "customer": {
        "id": "George_12",
        "name": null,
        "email": null,
        "phone": null,
        "phone_country_code": null
    },
    "description": "oda",
    "refunds": null,
    "disputes": null,
    "mandate_id": null,
    "mandate_data": null,
    "setup_future_usage": "off_session",
    "off_session": null,
    "capture_on": null,
    "capture_method": "automatic",
    "payment_method": "card",
    "payment_method_data": {
        "card": {
            "last4": "1091",
            "card_type": null,
            "card_network": null,
            "card_issuer": null,
            "card_issuing_country": null,
            "card_isin": "400000",
            "card_extended_bin": null,
            "card_exp_month": "01",
            "card_exp_year": "30",
            "card_holder_name": null,
            "payment_checks": {
                "description": "The street address and postal code matched.",
                "avs_result_code": "Y"
            },
            "authentication_data": null
        },
        "billing": null
    },
    "payment_token": null,
    "shipping": null,
    "billing": null,
    "order_details": null,
    "email": null,
    "name": null,
    "phone": null,
    "return_url": null,
    "authentication_type": "no_three_ds",
    "statement_descriptor_name": null,
    "statement_descriptor_suffix": null,
    "next_action": null,
    "cancellation_reason": null,
    "error_code": null,
    "error_message": null,
    "unified_code": null,
    "unified_message": null,
    "payment_experience": null,
    "payment_method_type": "credit",
    "connector_label": null,
    "business_country": null,
    "business_label": "default",
    "business_sub_label": null,
    "allowed_payment_method_types": null,
    "ephemeral_key": {
        "customer_id": "George_12",
        "created_at": 1757318307,
        "expires": 1757321907,
        "secret": "epk_acf8e718b07f4b3ba654f7e21aa06ec1"
    },
    "manual_retry_allowed": false,
    "connector_transaction_id": "120070562091",
    "frm_message": null,
    "metadata": null,
    "connector_metadata": null,
    "feature_metadata": {
        "redirect_response": null,
        "search_tags": null,
        "apple_pay_recurring_details": null,
        "gateway_system": "direct"
    },
    "reference_id": "120070562091",
    "payment_link": null,
    "profile_id": "pro_B1KJWgaKMyj9NJZOiGiL",
    "surcharge_details": null,
    "attempt_count": 1,
    "merchant_decision": null,
    "merchant_connector_id": "mca_1A9mFebPqoV963nNNKk4",
    "incremental_authorization_allowed": false,
    "authorization_count": null,
    "incremental_authorizations": null,
    "external_authentication_details": null,
    "external_3ds_authentication_attempted": false,
    "expires_on": "2025-09-08T08:13:27.079Z",
    "fingerprint": null,
    "browser_info": {
        "language": "nl-NL",
        "time_zone": 0,
        "ip_address": "13.232.74.226",
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36",
        "color_depth": 24,
        "java_enabled": true,
        "screen_width": 1536,
        "accept_header": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "screen_height": 723,
        "java_script_enabled": true
    },
    "payment_channel": null,
    "payment_method_id": "pm_8irM47OgNptHSjd4Yxjy",
    "network_transaction_id": "WME4UM2HEPYE12MVHMAXWRY",
    "payment_method_status": "active",
    "updated": "2025-09-08T07:58:29.279Z",
    "split_payments": null,
    "frm_metadata": null,
    "extended_authorization_applied": null,
    "capture_before": null,
    "merchant_order_reference_id": null,
    "order_tax_amount": null,
    "connector_mandate_id": "932579889-931872607",
    "card_discovery": "manual",
    "force_3ds_challenge": false,
    "force_3ds_challenge_trigger": false,
    "issuer_error_code": null,
    "issuer_error_message": null,
    "is_iframe_redirection_enabled": null,
    "whole_connector_response": null,
    "enable_partial_authorization": null
}
```

Create MIT

```
curl --location 'http://localhost:8080/payments' \
--header 'Content-Type: application/json' \
--header 'Accept: application/json' \
--header 'api-key: dev_QTV7hfwJvBw7V1Cw4IxHA8h1gg9in8dKYTNWtoWa6BkgfjvSr5r8Jg9gnZWL0wzX' \
--data '{
    "amount": 10000,
    "currency": "USD",
    "off_session": true,
    "confirm": true,
    "capture_method": "automatic",
    "recurring_details": {
        "type": "payment_method_id",
        "data": "pm_mfdEZEUD6TRHnl3xbXen"
    },
    
    "customer_id": "George_12",
    "connector": [
        "authorizedotnet"
    ]
}'
```
Response
```
{
    "payment_id": "pay_Uch1kSbBqSzglhlR9P3a",
    "merchant_id": "postman_merchant_GHAction_7ca00208-f80b-4f92-8ba9-3cbf9b0eb6ed",
    "status": "succeeded",
    "amount": 10000,
    "net_amount": 10000,
    "shipping_cost": null,
    "amount_capturable": 0,
    "amount_received": 10000,
    "connector": "authorizedotnet",
    "client_secret": "pay_Uch1kSbBqSzglhlR9P3a_secret_iskb1cwZZxOX0pJb3QgO",
    "created": "2025-09-08T07:59:30.857Z",
    "currency": "USD",
    "customer_id": "George_12",
    "customer": {
        "id": "George_12",
        "name": null,
        "email": null,
        "phone": null,
        "phone_country_code": null
    },
    "description": null,
    "refunds": null,
    "disputes": null,
    "mandate_id": null,
    "mandate_data": null,
    "setup_future_usage": null,
    "off_session": true,
    "capture_on": null,
    "capture_method": "automatic",
    "payment_method": "card",
    "payment_method_data": {
        "card": {
            "last4": "1091",
            "card_type": null,
            "card_network": null,
            "card_issuer": null,
            "card_issuing_country": null,
            "card_isin": "400000",
            "card_extended_bin": null,
            "card_exp_month": "01",
            "card_exp_year": "30",
            "card_holder_name": null,
            "payment_checks": null,
            "authentication_data": null
        },
        "billing": null
    },
    "payment_token": null,
    "shipping": null,
    "billing": null,
    "order_details": null,
    "email": null,
    "name": null,
    "phone": null,
    "return_url": null,
    "authentication_type": "no_three_ds",
    "statement_descriptor_name": null,
    "statement_descriptor_suffix": null,
    "next_action": null,
    "cancellation_reason": null,
    "error_code": null,
    "error_message": null,
    "unified_code": null,
    "unified_message": null,
    "payment_experience": null,
    "payment_method_type": "credit",
    "connector_label": null,
    "business_country": null,
    "business_label": "default",
    "business_sub_label": null,
    "allowed_payment_method_types": null,
    "ephemeral_key": {
        "customer_id": "George_12",
        "created_at": 1757318370,
        "expires": 1757321970,
        "secret": "epk_7b91847bfd94413c816e69a49135d0d0"
    },
    "manual_retry_allowed": false,
    "connector_transaction_id": "120070562151",
    "frm_message": null,
    "metadata": null,
    "connector_metadata": null,
    "feature_metadata": {
        "redirect_response": null,
        "search_tags": null,
        "apple_pay_recurring_details": null,
        "gateway_system": "direct"
    },
    "reference_id": "120070562151",
    "payment_link": null,
    "profile_id": "pro_B1KJWgaKMyj9NJZOiGiL",
    "surcharge_details": null,
    "attempt_count": 1,
    "merchant_decision": null,
    "merchant_connector_id": "mca_1A9mFebPqoV963nNNKk4",
    "incremental_authorization_allowed": false,
    "authorization_count": null,
    "incremental_authorizations": null,
    "external_authentication_details": null,
    "external_3ds_authentication_attempted": false,
    "expires_on": "2025-09-08T08:14:30.857Z",
    "fingerprint": null,
    "browser_info": null,
    "payment_channel": null,
    "payment_method_id": "pm_mfdEZEUD6TRHnl3xbXen",
    "network_transaction_id": "SOYOQ73IT3NFN1CB6VYKGWG",
    "payment_method_status": "active",
    "updated": "2025-09-08T07:59:31.707Z",
    "split_payments": null,
    "frm_metadata": null,
    "extended_authorization_applied": null,
    "capture_before": null,
    "merchant_order_reference_id": null,
    "order_tax_amount": null,
    "connector_mandate_id": "932579889-931872607",
    "card_discovery": "manual",
    "force_3ds_challenge": false,
    "force_3ds_challenge_trigger": false,
    "issuer_error_code": null,
    "issuer_error_message": null,
    "is_iframe_redirection_enabled": null,
    "whole_connector_response": null,
    "enable_partial_authorization": null
}
```

</details>

<details>
<summary> Create customer outside Hyperswitch and do a payment with the same customer in hyperswitch</summary>

```
curl --location 'https://apitest.authorize.net/xml/v1/request.api' \
--header 'Content-Type: application/json' \
--data-raw '{
    "createCustomerProfileRequest": {
        "merchantAuthentication": {
            "name": "",
            "transactionKey": ""
        },
     "profile": {
            "merchantCustomerId": "George_34",
            "email": "email16@here.com"
            // "description": "Profile description here"
            // "validationMode": "testMode",
        }
        
    }
}'
```

Response

```
{
    "customerProfileId": "932581492",
    "customerPaymentProfileIdList": [],
    "customerShippingAddressIdList": [],
    "validationDirectResponseList": [],
    "messages": {
        "resultCode": "Ok",
        "message": [
            {
                "code": "I00001",
                "text": "Successful."
            }
        ]
    }
}
```

Create a CIT payment with the same customer id and email

```
curl --location 'http://localhost:8080/payments' \
--header 'Content-Type: application/json' \
--header 'Accept: application/json' \
--header 'api-key: dev_ojXWyOIKxCbs6qJ1HUxnr0bRHtyqct6AmGj0rJJ6oxsFCsdmuGklbzGerp8MhttC' \
--data-raw '{
    "amount": 1,
    "currency": "USD",
    "confirm": true,
    "description": "oda",
    
    "capture_method": "automatic",

    
    "billing": {
       
        "address": {
            
            
            
            "city": "Ooty",
            "state": "TN",
            "zip": "02915",
            "country": "US",
            "first_name": "Mike",
            "last_name": "J. Hammer"
        }
    },
     "email": "email16@here.com",
    
    "customer_id": "George_34",
    "payment_method": "card",
    "payment_method_type": "credit",

    
    "payment_method_data": {
        "card": {
            "card_number": "4548817212493017",
            "card_exp_month": "01",
            "card_exp_year": "30",
            
            "card_cvc": "123"
            
        }
    },
    
    "customer_acceptance": {
        "acceptance_type": "online",
        "accepted_at": "2025-03-27T13:56:49.848Z",
        "online": {
            "ip_address": null,
            "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.3 Safari/605.1.15"
        }
    },
    "setup_future_usage": "off_session",
    "shipping": {
    "address": {
        "line1": "1467",
        "line2": "Harrison Street",
        "line3": "Harrison Street",
        "city": "San Fransico",
        "state": "North Carolina South",
        "zip": "94122",
        "country": "US",
        "first_name": "박성준",
        "last_name": "박성준"
    }},

    
    "browser_info": {
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36",
        "accept_header": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "language": "nl-NL",
        "color_depth": 24,
        "screen_height": 723,
        "screen_width": 1536,
        "time_zone": 0,
        "java_enabled": true,
        "java_script_enabled": true,
        "ip_address": "13.232.74.226"
    }
    
}'
```

Response
```
{
    "payment_id": "pay_7GK3Vi1p2X6nYTq0DNvi",
    "merchant_id": "postman_merchant_GHAction_f642d793-b987-4b8b-9d3d-760182ec59f5",
    "status": "succeeded",
    "amount": 1,
    "net_amount": 1,
    "shipping_cost": null,
    "amount_capturable": 0,
    "amount_received": 1,
    "connector": "authorizedotnet",
    "client_secret": "pay_7GK3Vi1p2X6nYTq0DNvi_secret_Mmie9JbCtArhQXrYdk02",
    "created": "2025-09-08T10:30:25.540Z",
    "currency": "USD",
    "customer_id": "George_34",
    "customer": {
        "id": "George_34",
        "name": null,
        "email": "email16@here.com",
        "phone": null,
        "phone_country_code": null
    },
    "description": "oda",
    "refunds": null,
    "disputes": null,
    "mandate_id": null,
    "mandate_data": null,
    "setup_future_usage": "off_session",
    "off_session": null,
    "capture_on": null,
    "capture_method": "automatic",
    "payment_method": "card",
    "payment_method_data": {
        "card": {
            "last4": "3017",
            "card_type": null,
            "card_network": null,
            "card_issuer": null,
            "card_issuing_country": null,
            "card_isin": "454881",
            "card_extended_bin": null,
            "card_exp_month": "01",
            "card_exp_year": "30",
            "card_holder_name": null,
            "payment_checks": {
                "description": "The street address and postal code matched.",
                "avs_result_code": "Y"
            },
            "authentication_data": null
        },
        "billing": null
    },
    "payment_token": null,
    "shipping": {
        "address": {
            "city": "San Fransico",
            "country": "US",
            "line1": "1467",
            "line2": "Harrison Street",
            "line3": "Harrison Street",
            "zip": "94122",
            "state": "North Carolina South",
            "first_name": "박성준",
            "last_name": "박성준",
            "origin_zip": null
        },
        "phone": null,
        "email": null
    },
    "billing": {
        "address": {
            "city": "Ooty",
            "country": "US",
            "line1": null,
            "line2": null,
            "line3": null,
            "zip": "02915",
            "state": "TN",
            "first_name": "Mike",
            "last_name": "J. Hammer",
            "origin_zip": null
        },
        "phone": null,
        "email": null
    },
    "order_details": null,
    "email": "email16@here.com",
    "name": null,
    "phone": null,
    "return_url": null,
    "authentication_type": "no_three_ds",
    "statement_descriptor_name": null,
    "statement_descriptor_suffix": null,
    "next_action": null,
    "cancellation_reason": null,
    "error_code": null,
    "error_message": null,
    "unified_code": null,
    "unified_message": null,
    "payment_experience": null,
    "payment_method_type": "credit",
    "connector_label": null,
    "business_country": null,
    "business_label": "default",
    "business_sub_label": null,
    "allowed_payment_method_types": null,
    "ephemeral_key": {
        "customer_id": "George_34",
        "created_at": 1757327425,
        "expires": 1757331025,
        "secret": "epk_745a154b3b4648e8888cd8959b6e1e31"
    },
    "manual_retry_allowed": false,
    "connector_transaction_id": "120070567790",
    "frm_message": null,
    "metadata": null,
    "connector_metadata": null,
    "feature_metadata": {
        "redirect_response": null,
        "search_tags": null,
        "apple_pay_recurring_details": null,
        "gateway_system": "direct"
    },
    "reference_id": "120070567790",
    "payment_link": null,
    "profile_id": "pro_4c5KStks69a6pdMZU2uv",
    "surcharge_details": null,
    "attempt_count": 1,
    "merchant_decision": null,
    "merchant_connector_id": "mca_1br5DZpuUOdTwIk619pY",
    "incremental_authorization_allowed": false,
    "authorization_count": null,
    "incremental_authorizations": null,
    "external_authentication_details": null,
    "external_3ds_authentication_attempted": false,
    "expires_on": "2025-09-08T10:45:25.540Z",
    "fingerprint": null,
    "browser_info": {
        "language": "nl-NL",
        "time_zone": 0,
        "ip_address": "13.232.74.226",
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36",
        "color_depth": 24,
        "java_enabled": true,
        "screen_width": 1536,
        "accept_header": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "screen_height": 723,
        "java_script_enabled": true
    },
    "payment_channel": null,
    "payment_method_id": "pm_kLwQZFTDZHyi7ZfsluUG",
    "network_transaction_id": "VQ5N75LVGWHYRFDV6SYXSN4",
    "payment_method_status": "active",
    "updated": "2025-09-08T10:30:27.640Z",
    "split_payments": null,
    "frm_metadata": null,
    "extended_authorization_applied": null,
    "capture_before": null,
    "merchant_order_reference_id": null,
    "order_tax_amount": null,
    "connector_mandate_id": "932581089-931874336",
    "card_discovery": "manual",
    "force_3ds_challenge": false,
    "force_3ds_challenge_trigger": false,
    "issuer_error_code": null,
    "issuer_error_message": null,
    "is_iframe_redirection_enabled": null,
    "whole_connector_response": null,
    "enable_partial_authorization": null
}
```

MIT transaction

```
curl --location 'http://localhost:8080/payments' \
--header 'Content-Type: application/json' \
--header 'Accept: application/json' \
--header 'api-key: dev_ojXWyOIKxCbs6qJ1HUxnr0bRHtyqct6AmGj0rJJ6oxsFCsdmuGklbzGerp8MhttC' \
--data '{
    "amount": 10000,
    "currency": "USD",
    "off_session": true,
    "confirm": true,
    "capture_method": "automatic",
    "recurring_details": {
        "type": "payment_method_id",
        "data": "pm_kLwQZFTDZHyi7ZfsluUG"
    },
    
    "customer_id": "George_34",
    "connector": [
        "authorizedotnet"
    ]
}'
```

Response
```
{
    "payment_id": "pay_gMzzuKqBB2SpmylOck8P",
    "merchant_id": "postman_merchant_GHAction_f642d793-b987-4b8b-9d3d-760182ec59f5",
    "status": "succeeded",
    "amount": 10000,
    "net_amount": 10000,
    "shipping_cost": null,
    "amount_capturable": 0,
    "amount_received": 10000,
    "connector": "authorizedotnet",
    "client_secret": "pay_gMzzuKqBB2SpmylOck8P_secret_lB92QYHhKDPHvp4wNJNJ",
    "created": "2025-09-08T10:32:15.961Z",
    "currency": "USD",
    "customer_id": "George_34",
    "customer": {
        "id": "George_34",
        "name": null,
        "email": "email16@here.com",
        "phone": null,
        "phone_country_code": null
    },
    "description": null,
    "refunds": null,
    "disputes": null,
    "mandate_id": null,
    "mandate_data": null,
    "setup_future_usage": null,
    "off_session": true,
    "capture_on": null,
    "capture_method": "automatic",
    "payment_method": "card",
    "payment_method_data": {
        "card": {
            "last4": "3017",
            "card_type": null,
            "card_network": null,
            "card_issuer": null,
            "card_issuing_country": null,
            "card_isin": "454881",
            "card_extended_bin": null,
            "card_exp_month": "01",
            "card_exp_year": "30",
            "card_holder_name": "Mike J. Hammer",
            "payment_checks": null,
            "authentication_data": null
        },
        "billing": null
    },
    "payment_token": null,
    "shipping": null,
    "billing": null,
    "order_details": null,
    "email": "email16@here.com",
    "name": null,
    "phone": null,
    "return_url": null,
    "authentication_type": "no_three_ds",
    "statement_descriptor_name": null,
    "statement_descriptor_suffix": null,
    "next_action": null,
    "cancellation_reason": null,
    "error_code": null,
    "error_message": null,
    "unified_code": null,
    "unified_message": null,
    "payment_experience": null,
    "payment_method_type": "credit",
    "connector_label": null,
    "business_country": null,
    "business_label": "default",
    "business_sub_label": null,
    "allowed_payment_method_types": null,
    "ephemeral_key": {
        "customer_id": "George_34",
        "created_at": 1757327535,
        "expires": 1757331135,
        "secret": "epk_9a461e1f7e0243a1b8c0c896a5928c21"
    },
    "manual_retry_allowed": false,
    "connector_transaction_id": "120070567859",
    "frm_message": null,
    "metadata": null,
    "connector_metadata": null,
    "feature_metadata": {
        "redirect_response": null,
        "search_tags": null,
        "apple_pay_recurring_details": null,
        "gateway_system": "direct"
    },
    "reference_id": "120070567859",
    "payment_link": null,
    "profile_id": "pro_4c5KStks69a6pdMZU2uv",
    "surcharge_details": null,
    "attempt_count": 1,
    "merchant_decision": null,
    "merchant_connector_id": "mca_1br5DZpuUOdTwIk619pY",
    "incremental_authorization_allowed": false,
    "authorization_count": null,
    "incremental_authorizations": null,
    "external_authentication_details": null,
    "external_3ds_authentication_attempted": false,
    "expires_on": "2025-09-08T10:47:15.961Z",
    "fingerprint": null,
    "browser_info": null,
    "payment_channel": null,
    "payment_method_id": "pm_kLwQZFTDZHyi7ZfsluUG",
    "network_transaction_id": "Z7KZ0F7C23FF7OMWVWMZK2B",
    "payment_method_status": "active",
    "updated": "2025-09-08T10:32:18.018Z",
    "split_payments": null,
    "frm_metadata": null,
    "extended_authorization_applied": null,
    "capture_before": null,
    "merchant_order_reference_id": null,
    "order_tax_amount": null,
    "connector_mandate_id": "932581089-931874336",
    "card_discovery": "manual",
    "force_3ds_challenge": false,
    "force_3ds_challenge_trigger": false,
    "issuer_error_code": null,
    "issuer_error_message": null,
    "is_iframe_redirection_enabled": null,
    "whole_connector_response": null,
    "enable_partial_authorization": null
}
```
</details>
<details>
<summary> Sanity checks </summary>

Create a normal payment with customer id

```
curl --location 'http://localhost:8080/payments' \
--header 'Content-Type: application/json' \
--header 'Accept: application/json' \
--header 'api-key: dev_ojXWyOIKxCbs6qJ1HUxnr0bRHtyqct6AmGj0rJJ6oxsFCsdmuGklbzGerp8MhttC' \
--data-raw '{
    "amount": 1,
    "currency": "USD",
    "confirm": true,
    "description": "oda",
    
    "capture_method": "automatic",
    
    
    
    
    
    
    
    "billing": {
       
        "address": {
            
            
            
            "city": "Ooty",
            "state": "TN",
            "zip": "02915",
            "country": "US",
            "first_name": "Mike",
            "last_name": "J. Hammer"
        }
    },
     "email": "email16@here.com",
    
    "customer_id": "George_39",
    "payment_method": "card",
    "payment_method_type": "credit",

    "payment_method_data": {
        "card": {
            "card_number": "4548817212493017",
            "card_exp_month": "01",
            "card_exp_year": "30",
            
            "card_cvc": "123"
            
        }
    },

    "shipping": {
    "address": {
        "line1": "1467",
        "line2": "Harrison Street",
        "line3": "Harrison Street",
        "city": "San Fransico",
        "state": "North Carolina South",
        "zip": "94122",
        "country": "US",
        "first_name": "박성준",
        "last_name": "박성준"
    }},
    

    
    "browser_info": {
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36",
        "accept_header": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "language": "nl-NL",
        "color_depth": 24,
        "screen_height": 723,
        "screen_width": 1536,
        "time_zone": 0,
        "java_enabled": true,
        "java_script_enabled": true,
        "ip_address": "13.232.74.226"
    }
    
    
}'
```

Response
```
{
    "payment_id": "pay_F6FBGXBosUnkMWVCwzwC",
    "merchant_id": "postman_merchant_GHAction_f642d793-b987-4b8b-9d3d-760182ec59f5",
    "status": "succeeded",
    "amount": 1,
    "net_amount": 1,
    "shipping_cost": null,
    "amount_capturable": 0,
    "amount_received": 1,
    "connector": "authorizedotnet",
    "client_secret": "pay_F6FBGXBosUnkMWVCwzwC_secret_WL7XVnHG4iF7XUybRSw0",
    "created": "2025-09-08T10:35:58.529Z",
    "currency": "USD",
    "customer_id": "George_39",
    "customer": {
        "id": "George_39",
        "name": null,
        "email": "email16@here.com",
        "phone": null,
        "phone_country_code": null
    },
    "description": "oda",
    "refunds": null,
    "disputes": null,
    "mandate_id": null,
    "mandate_data": null,
    "setup_future_usage": null,
    "off_session": null,
    "capture_on": null,
    "capture_method": "automatic",
    "payment_method": "card",
    "payment_method_data": {
        "card": {
            "last4": "3017",
            "card_type": null,
            "card_network": null,
            "card_issuer": null,
            "card_issuing_country": null,
            "card_isin": "454881",
            "card_extended_bin": null,
            "card_exp_month": "01",
            "card_exp_year": "30",
            "card_holder_name": null,
            "payment_checks": {
                "description": "The street address and postal code matched.",
                "avs_result_code": "Y"
            },
            "authentication_data": null
        },
        "billing": null
    },
    "payment_token": null,
    "shipping": {
        "address": {
            "city": "San Fransico",
            "country": "US",
            "line1": "1467",
            "line2": "Harrison Street",
            "line3": "Harrison Street",
            "zip": "94122",
            "state": "North Carolina South",
            "first_name": "박성준",
            "last_name": "박성준",
            "origin_zip": null
        },
        "phone": null,
        "email": null
    },
    "billing": {
        "address": {
            "city": "Ooty",
            "country": "US",
            "line1": null,
            "line2": null,
            "line3": null,
            "zip": "02915",
            "state": "TN",
            "first_name": "Mike",
            "last_name": "J. Hammer",
            "origin_zip": null
        },
        "phone": null,
        "email": null
    },
    "order_details": null,
    "email": "email16@here.com",
    "name": null,
    "phone": null,
    "return_url": null,
    "authentication_type": "no_three_ds",
    "statement_descriptor_name": null,
    "statement_descriptor_suffix": null,
    "next_action": null,
    "cancellation_reason": null,
    "error_code": null,
    "error_message": null,
    "unified_code": null,
    "unified_message": null,
    "payment_experience": null,
    "payment_method_type": "credit",
    "connector_label": null,
    "business_country": null,
    "business_label": "default",
    "business_sub_label": null,
    "allowed_payment_method_types": null,
    "ephemeral_key": {
        "customer_id": "George_39",
        "created_at": 1757327758,
        "expires": 1757331358,
        "secret": "epk_eacacf5bc45041b1abcae4a96930bada"
    },
    "manual_retry_allowed": false,
    "connector_transaction_id": "120070568055",
    "frm_message": null,
    "metadata": null,
    "connector_metadata": null,
    "feature_metadata": {
        "redirect_response": null,
        "search_tags": null,
        "apple_pay_recurring_details": null,
        "gateway_system": "direct"
    },
    "reference_id": "120070568055",
    "payment_link": null,
    "profile_id": "pro_4c5KStks69a6pdMZU2uv",
    "surcharge_details": null,
    "attempt_count": 1,
    "merchant_decision": null,
    "merchant_connector_id": "mca_1br5DZpuUOdTwIk619pY",
    "incremental_authorization_allowed": false,
    "authorization_count": null,
    "incremental_authorizations": null,
    "external_authentication_details": null,
    "external_3ds_authentication_attempted": false,
    "expires_on": "2025-09-08T10:50:58.528Z",
    "fingerprint": null,
    "browser_info": {
        "language": "nl-NL",
        "time_zone": 0,
        "ip_address": "13.232.74.226",
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36",
        "color_depth": 24,
        "java_enabled": true,
        "screen_width": 1536,
        "accept_header": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "screen_height": 723,
        "java_script_enabled": true
    },
    "payment_channel": null,
    "payment_method_id": null,
    "network_transaction_id": "HOADKZPWOV4VMD7463CTMDE",
    "payment_method_status": null,
    "updated": "2025-09-08T10:36:01.130Z",
    "split_payments": null,
    "frm_metadata": null,
    "extended_authorization_applied": null,
    "capture_before": null,
    "merchant_order_reference_id": null,
    "order_tax_amount": null,
    "connector_mandate_id": null,
    "card_discovery": "manual",
    "force_3ds_challenge": false,
    "force_3ds_challenge_trigger": false,
    "issuer_error_code": null,
    "issuer_error_message": null,
    "is_iframe_redirection_enabled": null,
    "whole_connector_response": null,
    "enable_partial_authorization": null
}
```
>Note: Check that the customer id is present in the authorize.net dashboard

Payment without customer id 

```
curl --location 'http://localhost:8080/payments' \
--header 'Content-Type: application/json' \
--header 'Accept: application/json' \
--header 'api-key: dev_ojXWyOIKxCbs6qJ1HUxnr0bRHtyqct6AmGj0rJJ6oxsFCsdmuGklbzGerp8MhttC' \
--data '{
    "amount": 1,
    "currency": "USD",
    "confirm": true,
    "description": "oda",
    
    "capture_method": "automatic",

    "billing": {
       
        "address": {
            
            
            
            "city": "Ooty",
            "state": "TN",
            "zip": "02915",
            "country": "US",
            "first_name": "Mike",
            "last_name": "J. Hammer"
        }
    },

    
    "payment_method": "card",
    "payment_method_type": "credit",

    "payment_method_data": {
        "card": {
            "card_number": "4548817212493017",
            "card_exp_month": "01",
            "card_exp_year": "30",
            
            "card_cvc": "123"
            
        }
    },

    
    "shipping": {
    "address": {
        "line1": "1467",
        "line2": "Harrison Street",
        "line3": "Harrison Street",
        "city": "San Fransico",
        "state": "North Carolina South",
        "zip": "94122",
        "country": "US",
        "first_name": "박성준",
        "last_name": "박성준"
    }},
    

    
    
    "browser_info": {
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36",
        "accept_header": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "language": "nl-NL",
        "color_depth": 24,
        "screen_height": 723,
        "screen_width": 1536,
        "time_zone": 0,
        "java_enabled": true,
        "java_script_enabled": true,
        "ip_address": "13.232.74.226"
    }

    
}'
```

Response

```
{
    "payment_id": "pay_Q45X5L5EevrqOMYCOCZ0",
    "merchant_id": "postman_merchant_GHAction_f642d793-b987-4b8b-9d3d-760182ec59f5",
    "status": "succeeded",
    "amount": 1,
    "net_amount": 1,
    "shipping_cost": null,
    "amount_capturable": 0,
    "amount_received": 1,
    "connector": "authorizedotnet",
    "client_secret": "pay_Q45X5L5EevrqOMYCOCZ0_secret_pLPOQD9Tsy81KfFjKB6X",
    "created": "2025-09-08T10:37:40.898Z",
    "currency": "USD",
    "customer_id": null,
    "customer": null,
    "description": "oda",
    "refunds": null,
    "disputes": null,
    "mandate_id": null,
    "mandate_data": null,
    "setup_future_usage": null,
    "off_session": null,
    "capture_on": null,
    "capture_method": "automatic",
    "payment_method": "card",
    "payment_method_data": {
        "card": {
            "last4": "3017",
            "card_type": null,
            "card_network": null,
            "card_issuer": null,
            "card_issuing_country": null,
            "card_isin": "454881",
            "card_extended_bin": null,
            "card_exp_month": "01",
            "card_exp_year": "30",
            "card_holder_name": null,
            "payment_checks": {
                "description": "The street address and postal code matched.",
                "avs_result_code": "Y"
            },
            "authentication_data": null
        },
        "billing": null
    },
    "payment_token": null,
    "shipping": {
        "address": {
            "city": "San Fransico",
            "country": "US",
            "line1": "1467",
            "line2": "Harrison Street",
            "line3": "Harrison Street",
            "zip": "94122",
            "state": "North Carolina South",
            "first_name": "박성준",
            "last_name": "박성준",
            "origin_zip": null
        },
        "phone": null,
        "email": null
    },
    "billing": {
        "address": {
            "city": "Ooty",
            "country": "US",
            "line1": null,
            "line2": null,
            "line3": null,
            "zip": "02915",
            "state": "TN",
            "first_name": "Mike",
            "last_name": "J. Hammer",
            "origin_zip": null
        },
        "phone": null,
        "email": null
    },
    "order_details": null,
    "email": null,
    "name": null,
    "phone": null,
    "return_url": null,
    "authentication_type": "no_three_ds",
    "statement_descriptor_name": null,
    "statement_descriptor_suffix": null,
    "next_action": null,
    "cancellation_reason": null,
    "error_code": null,
    "error_message": null,
    "unified_code": null,
    "unified_message": null,
    "payment_experience": null,
    "payment_method_type": "credit",
    "connector_label": null,
    "business_country": null,
    "business_label": "default",
    "business_sub_label": null,
    "allowed_payment_method_types": null,
    "ephemeral_key": null,
    "manual_retry_allowed": false,
    "connector_transaction_id": "120070568151",
    "frm_message": null,
    "metadata": null,
    "connector_metadata": null,
    "feature_metadata": {
        "redirect_response": null,
        "search_tags": null,
        "apple_pay_recurring_details": null,
        "gateway_system": "direct"
    },
    "reference_id": "120070568151",
    "payment_link": null,
    "profile_id": "pro_4c5KStks69a6pdMZU2uv",
    "surcharge_details": null,
    "attempt_count": 1,
    "merchant_decision": null,
    "merchant_connector_id": "mca_1br5DZpuUOdTwIk619pY",
    "incremental_authorization_allowed": false,
    "authorization_count": null,
    "incremental_authorizations": null,
    "external_authentication_details": null,
    "external_3ds_authentication_attempted": false,
    "expires_on": "2025-09-08T10:52:40.898Z",
    "fingerprint": null,
    "browser_info": {
        "language": "nl-NL",
        "time_zone": 0,
        "ip_address": "13.232.74.226",
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36",
        "color_depth": 24,
        "java_enabled": true,
        "screen_width": 1536,
        "accept_header": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "screen_height": 723,
        "java_script_enabled": true
    },
    "payment_channel": null,
    "payment_method_id": null,
    "network_transaction_id": "OB29UI4CXX0LM51T4RT6F6Z",
    "payment_method_status": null,
    "updated": "2025-09-08T10:37:43.437Z",
    "split_payments": null,
    "frm_metadata": null,
    "extended_authorization_applied": null,
    "capture_before": null,
    "merchant_order_reference_id": null,
    "order_tax_amount": null,
    "connector_mandate_id": null,
    "card_discovery": "manual",
    "force_3ds_challenge": false,
    "force_3ds_challenge_trigger": false,
    "issuer_error_code": null,
    "issuer_error_message": null,
    "is_iframe_redirection_enabled": null,
    "whole_connector_response": null,
    "enable_partial_authorization": null
}
```
</details> 

<details>
<summary>Wallet test - Googlepay</summary>

Create a normal payment with customer id, customer id should be shown for the transaction

```
curl --location 'http://localhost:8080/payments' \
--header 'Content-Type: application/json' \
--header 'Accept: application/json' \
--header 'api-key: dev_KE9tLklSMgmO0Wh3Zy8zu5IoKEEKynF16mY53eV84Dx5TbRjpHfSJuUt7WVEgd4r' \
--data '{
    "amount": 100,
    "currency": "USD",
    "confirm": true,
    "description": "oda",

    
    "return_url": "https://app.hyperswitch.io/dashboard/sdk",
    "payment_method": "wallet",
    "payment_method_type": "google_pay",
    "payment_method_data": {
        "wallet": {
            "google_pay": {
                "type": "CARD",
                "description": "Test Card: Visa •••• 1111",
                "info": {
                    "assurance_details": {
                        "account_verified": true,
                        "card_holder_authenticated": false
                    },
                    "card_details": "1111",
                    "card_funding_source": "CREDIT",
                    "card_network": "VISA"
                },
                "tokenization_data": {
                    "token": "{\"signature\":\"MEUCI**************",
                    "type": "PAYMENT_GATEWAY"
                }
            }
        }
    },
    "connector": [
        "authorizedotnet"
    ],
    "browser_info": {
        "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
        "accept_header": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "language": "en-GB",
        "color_depth": 24,
        "screen_height": 1117,
        "screen_width": 1728,
        "time_zone": -330,
        "java_enabled": true,
        "java_script_enabled": true,
        "device_model": "Macintosh",
        "os_type": "macOS",
        "os_version": "10.15.7"
    },
    "customer_id": "cus_12345"
    
}'
```

Response

```
{
    "payment_id": "pay_LDMqfa273PAHeseOz0As",
    "merchant_id": "postman_merchant_GHAction_5c96aafd-4990-44b3-b44d-c66b2a76f962",
    "status": "succeeded",
    "amount": 100,
    "net_amount": 100,
    "shipping_cost": null,
    "amount_capturable": 0,
    "amount_received": 100,
    "connector": "authorizedotnet",
    "client_secret": "pay_LDMqfa273PAHeseOz0As_secret_I4SZIKuOKqNufNurB1mz",
    "created": "2025-09-08T12:06:57.848Z",
    "currency": "USD",
    "customer_id": "cus_12345",
    "customer": {
        "id": "cus_12345",
        "name": null,
        "email": null,
        "phone": null,
        "phone_country_code": null
    },
    "description": "oda",
    "refunds": null,
    "disputes": null,
    "mandate_id": null,
    "mandate_data": null,
    "setup_future_usage": null,
    "off_session": null,
    "capture_on": null,
    "capture_method": null,
    "payment_method": "wallet",
    "payment_method_data": {
        "wallet": {
            "google_pay": {
                "last4": "1111",
                "card_network": "VISA",
                "type": "CARD"
            }
        },
        "billing": null
    },
    "payment_token": null,
    "shipping": null,
    "billing": null,
    "order_details": null,
    "email": null,
    "name": null,
    "phone": null,
    "return_url": "https://app.hyperswitch.io/dashboard/sdk",
    "authentication_type": "three_ds",
    "statement_descriptor_name": null,
    "statement_descriptor_suffix": null,
    "next_action": null,
    "cancellation_reason": null,
    "error_code": null,
    "error_message": null,
    "unified_code": null,
    "unified_message": null,
    "payment_experience": null,
    "payment_method_type": "google_pay",
    "connector_label": null,
    "business_country": null,
    "business_label": "default",
    "business_sub_label": null,
    "allowed_payment_method_types": null,
    "ephemeral_key": {
        "customer_id": "cus_12345",
        "created_at": 1757333217,
        "expires": 1757336817,
        "secret": "epk_07214c9b7ff54164bd6c26ecf9895f2a"
    },
    "manual_retry_allowed": false,
    "connector_transaction_id": "80044364289",
    "frm_message": null,
    "metadata": null,
    "connector_metadata": null,
    "feature_metadata": {
        "redirect_response": null,
        "search_tags": null,
        "apple_pay_recurring_details": null,
        "gateway_system": "direct"
    },
    "reference_id": "80044364289",
    "payment_link": null,
    "profile_id": "pro_ayYGLzXmcF3E9JgC7HZO",
    "surcharge_details": null,
    "attempt_count": 1,
    "merchant_decision": null,
    "merchant_connector_id": "mca_EnqpwnxfB7GppsknYX7r",
    "incremental_authorization_allowed": false,
    "authorization_count": null,
    "incremental_authorizations": null,
    "external_authentication_details": null,
    "external_3ds_authentication_attempted": false,
    "expires_on": "2025-09-08T12:21:57.848Z",
    "fingerprint": null,
    "browser_info": {
        "os_type": "macOS",
        "language": "en-GB",
        "time_zone": -330,
        "os_version": "10.15.7",
        "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
        "color_depth": 24,
        "device_model": "Macintosh",
        "java_enabled": true,
        "screen_width": 1728,
        "accept_header": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "screen_height": 1117,
        "java_script_enabled": true
    },
    "payment_channel": null,
    "payment_method_id": null,
    "network_transaction_id": "XPSPFFQOROVN42C5WFHQEM2",
    "payment_method_status": null,
    "updated": "2025-09-08T12:07:00.691Z",
    "split_payments": null,
    "frm_metadata": null,
    "extended_authorization_applied": null,
    "capture_before": null,
    "merchant_order_reference_id": null,
    "order_tax_amount": null,
    "connector_mandate_id": null,
    "card_discovery": null,
    "force_3ds_challenge": false,
    "force_3ds_challenge_trigger": false,
    "issuer_error_code": null,
    "issuer_error_message": null,
    "is_iframe_redirection_enabled": null,
    "whole_connector_response": null,
    "enable_partial_authorization": null
}
```

Create Cit twice for same customer id and same payment method and create trigger an MIT

CIT  1
```
curl --location 'http://localhost:8080/payments' \
--header 'Content-Type: application/json' \
--header 'Accept: application/json' \
--header 'api-key: dev_KE9tLklSMgmO0Wh3Zy8zu5IoKEEKynF16mY53eV84Dx5TbRjpHfSJuUt7WVEgd4r' \
--data '{
    "amount": 100,
    "currency": "USD",
    "confirm": true,
    "description": "oda",
    

    
    "return_url": "https://app.hyperswitch.io/dashboard/sdk",
    "payment_method": "wallet",
    "payment_method_type": "google_pay",
    "payment_method_data": {
        "wallet": {
            "google_pay": {
                "type": "CARD",
                "description": "Test Card: Visa •••• 1111",
                "info": {
                    "assurance_details": {
                        "account_verified": true,
                        "card_holder_authenticated": false
                    },
                    "card_details": "1111",
                    "card_funding_source": "CREDIT",
                    "card_network": "VISA"
                },
                "tokenization_data": {
                    "token": "{\"signature\":\"MEUCIQD8rFtn1qHNMINks9KByeBWj6IlMDTNlkXHXvfi4ucVlgIgS2ouIBMSCILZe3YsxWeUpsGK5sDF70zKWIkL3xWzqwY\\u003d\",\"protocolVersion\":\"ECv1\",\"signedMessage\":\"{\\\"encryptedMessage\\\":\\\"wIbdKS71cuojES2JxwMQW54T2M//izzeBHj8hJUpi0ExBsrjc/OUe6XhHRopgmB+VQjfJhObfNH+uHlXypZREefBPlH78wY8Bg5s3t6ufWGXLTQQGsfhoufKbJnZrshrxYUti8+6CByH62uNq8oxCp+XuTCWnXab8aLgGWyY5wDxtC9K/1wzN6PkPPyFDNwL6jEygIEr4rN9N4zTzcoLVJRqavqa19d2z26Ley6p7NNfLFZMjczuBdyRWSj7B5zTNT7WUgpF0KUiRvdMYELGuD1R93MeBsazJS5WJD*****************"}\"}",
                    "type": "PAYMENT_GATEWAY"
                }
            }
        }
    },
    "connector": [
        "authorizedotnet"
    ],
    "browser_info": {
        "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
        "accept_header": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "language": "en-GB",
        "color_depth": 24,
        "screen_height": 1117,
        "screen_width": 1728,
        "time_zone": -330,
        "java_enabled": true,
        "java_script_enabled": true,
        "device_model": "Macintosh",
        "os_type": "macOS",
        "os_version": "10.15.7"
    },
    "customer_id": "cus_12345",
        "customer_acceptance": {
        "acceptance_type": "online",
        "accepted_at": "2025-03-27T13:56:49.848Z",
        "online": {
            "ip_address": null,
            "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.3 Safari/605.1.15"
        }
    },
    "setup_future_usage": "off_session"
    
}'
```
Response

```
{
    "payment_id": "pay_37NcwVUnk2TRXeWWSffF",
    "merchant_id": "postman_merchant_GHAction_5c96aafd-4990-44b3-b44d-c66b2a76f962",
    "status": "succeeded",
    "amount": 100,
    "net_amount": 100,
    "shipping_cost": null,
    "amount_capturable": 0,
    "amount_received": 100,
    "connector": "authorizedotnet",
    "client_secret": "pay_37NcwVUnk2TRXeWWSffF_secret_nGC20xDeGX7b3rfUUBLv",
    "created": "2025-09-08T12:19:11.685Z",
    "currency": "USD",
    "customer_id": "cus_12345",
    "customer": {
        "id": "cus_12345",
        "name": null,
        "email": null,
        "phone": null,
        "phone_country_code": null
    },
    "description": "oda",
    "refunds": null,
    "disputes": null,
    "mandate_id": null,
    "mandate_data": null,
    "setup_future_usage": "off_session",
    "off_session": null,
    "capture_on": null,
    "capture_method": null,
    "payment_method": "wallet",
    "payment_method_data": {
        "wallet": {
            "google_pay": {
                "last4": "1111",
                "card_network": "VISA",
                "type": "CARD"
            }
        },
        "billing": null
    },
    "payment_token": null,
    "shipping": null,
    "billing": null,
    "order_details": null,
    "email": null,
    "name": null,
    "phone": null,
    "return_url": "https://app.hyperswitch.io/dashboard/sdk",
    "authentication_type": "three_ds",
    "statement_descriptor_name": null,
    "statement_descriptor_suffix": null,
    "next_action": null,
    "cancellation_reason": null,
    "error_code": null,
    "error_message": null,
    "unified_code": null,
    "unified_message": null,
    "payment_experience": null,
    "payment_method_type": "google_pay",
    "connector_label": null,
    "business_country": null,
    "business_label": "default",
    "business_sub_label": null,
    "allowed_payment_method_types": null,
    "ephemeral_key": {
        "customer_id": "cus_12345",
        "created_at": 1757333951,
        "expires": 1757337551,
        "secret": "epk_237bd58adf8d48cb9ad6d38fdcdc3b07"
    },
    "manual_retry_allowed": false,
    "connector_transaction_id": "80044364497",
    "frm_message": null,
    "metadata": null,
    "connector_metadata": null,
    "feature_metadata": {
        "redirect_response": null,
        "search_tags": null,
        "apple_pay_recurring_details": null,
        "gateway_system": "direct"
    },
    "reference_id": "80044364497",
    "payment_link": null,
    "profile_id": "pro_ayYGLzXmcF3E9JgC7HZO",
    "surcharge_details": null,
    "attempt_count": 1,
    "merchant_decision": null,
    "merchant_connector_id": "mca_EnqpwnxfB7GppsknYX7r",
    "incremental_authorization_allowed": false,
    "authorization_count": null,
    "incremental_authorizations": null,
    "external_authentication_details": null,
    "external_3ds_authentication_attempted": false,
    "expires_on": "2025-09-08T12:34:11.685Z",
    "fingerprint": null,
    "browser_info": {
        "os_type": "macOS",
        "language": "en-GB",
        "time_zone": -330,
        "os_version": "10.15.7",
        "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
        "color_depth": 24,
        "device_model": "Macintosh",
        "java_enabled": true,
        "screen_width": 1728,
        "accept_header": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "screen_height": 1117,
        "java_script_enabled": true
    },
    "payment_channel": null,
    "payment_method_id": "pm_49DfXLBrPJ1t90oFfmPi",
    "network_transaction_id": "BHYELB7L0V5C5Q4NW72AWX0",
    "payment_method_status": "active",
    "updated": "2025-09-08T12:19:13.552Z",
    "split_payments": null,
    "frm_metadata": null,
    "extended_authorization_applied": null,
    "capture_before": null,
    "merchant_order_reference_id": null,
    "order_tax_amount": null,
    "connector_mandate_id": "524439809-536594893",
    "card_discovery": null,
    "force_3ds_challenge": false,
    "force_3ds_challenge_trigger": false,
    "issuer_error_code": null,
    "issuer_error_message": null,
    "is_iframe_redirection_enabled": null,
    "whole_connector_response": null,
    "enable_partial_authorization": null
}
```

CIT 2

```
curl --location 'http://localhost:8080/payments' \
--header 'Content-Type: application/json' \
--header 'Accept: application/json' \
--header 'api-key: dev_KE9tLklSMgmO0Wh3Zy8zu5IoKEEK******************************' \
--data '{
    "amount": 100,
    "currency": "USD",
    "confirm": true,
    "description": "oda",
    "return_url": "https://app.hyperswitch.io/dashboard/sdk",
    "payment_method": "wallet",
    "payment_method_type": "google_pay",
    "payment_method_data": {
        "wallet": {
            "google_pay": {
                "type": "CARD",
                "description": "Test Card: Visa •••• 1111",
                "info": {
                    "assurance_details": {
                        "account_verified": true,
                        "card_holder_authenticated": false
                    },
                    "card_details": "1111",
                    "card_funding_source": "CREDIT",
                    "card_network": "VISA"
                },
                "tokenization_data": {
                    "token": "{\"signature\":\"MEUCIQD8rFtn1qHNMINks9KByeBWj6IlMDTNlkXHXvfi4ucVlgIgS2ouIBMSCILZe3YsxWeUpsGK5sDF70zKWIkL3xWzqwY\\u003d\",\"protocolVersion\":\"ECv1\",\"signedMessage\":\"{\\\"encryptedMessage\\\":\\\"wIbdKS71cuojES2JxwMQW54T2M//izzeBHj8hJUpi0ExBsrjc/OUe6XhHRopgmB+VQjfJhObfNH+uHlXypZREefBPlH78wY8Bg5s3t6u****************************\"}\"}",
                    "type": "PAYMENT_GATEWAY"
                }
            }
        }
    },
    "connector": [
        "authorizedotnet"
    ],
    "browser_info": {
        "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
        "accept_header": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "language": "en-GB",
        "color_depth": 24,
        "screen_height": 1117,
        "screen_width": 1728,
        "time_zone": -330,
        "java_enabled": true,
        "java_script_enabled": true,
        "device_model": "Macintosh",
        "os_type": "macOS",
        "os_version": "10.15.7"
    },
    "customer_id": "cus_12345",
        "customer_acceptance": {
        "acceptance_type": "online",
        "accepted_at": "2025-03-27T13:56:49.848Z",
        "online": {
            "ip_address": null,
            "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.3 Safari/605.1.15"
        }
    },
    "setup_future_usage": "off_session"
    
}'
```
Response

```
{
    "payment_id": "pay_FhZxwJ66xE2c3iNTmcjw",
    "merchant_id": "postman_merchant_GHAction_5c96aafd-4990-44b3-b44d-c66b2a76f962",
    "status": "succeeded",
    "amount": 100,
    "net_amount": 100,
    "shipping_cost": null,
    "amount_capturable": 0,
    "amount_received": 100,
    "connector": "authorizedotnet",
    "client_secret": "pay_FhZxwJ66xE2c3iNTmcjw_secret_EGOtM2GS2cXoBn7byX9K",
    "created": "2025-09-08T12:20:27.430Z",
    "currency": "USD",
    "customer_id": "cus_12345",
    "customer": {
        "id": "cus_12345",
        "name": null,
        "email": null,
        "phone": null,
        "phone_country_code": null
    },
    "description": "oda",
    "refunds": null,
    "disputes": null,
    "mandate_id": null,
    "mandate_data": null,
    "setup_future_usage": "off_session",
    "off_session": null,
    "capture_on": null,
    "capture_method": null,
    "payment_method": "wallet",
    "payment_method_data": {
        "wallet": {
            "google_pay": {
                "last4": "1111",
                "card_network": "VISA",
                "type": "CARD"
            }
        },
        "billing": null
    },
    "payment_token": null,
    "shipping": null,
    "billing": null,
    "order_details": null,
    "email": null,
    "name": null,
    "phone": null,
    "return_url": "https://app.hyperswitch.io/dashboard/sdk",
    "authentication_type": "three_ds",
    "statement_descriptor_name": null,
    "statement_descriptor_suffix": null,
    "next_action": null,
    "cancellation_reason": null,
    "error_code": null,
    "error_message": null,
    "unified_code": null,
    "unified_message": null,
    "payment_experience": null,
    "payment_method_type": "google_pay",
    "connector_label": null,
    "business_country": null,
    "business_label": "default",
    "business_sub_label": null,
    "allowed_payment_method_types": null,
    "ephemeral_key": {
        "customer_id": "cus_12345",
        "created_at": 1757334027,
        "expires": 1757337627,
        "secret": "epk_d387f51d21cb4643bef5d6a22a5dc38e"
    },
    "manual_retry_allowed": false,
    "connector_transaction_id": "80044364526",
    "frm_message": null,
    "metadata": null,
    "connector_metadata": null,
    "feature_metadata": {
        "redirect_response": null,
        "search_tags": null,
        "apple_pay_recurring_details": null,
        "gateway_system": "direct"
    },
    "reference_id": "80044364526",
    "payment_link": null,
    "profile_id": "pro_ayYGLzXmcF3E9JgC7HZO",
    "surcharge_details": null,
    "attempt_count": 1,
    "merchant_decision": null,
    "merchant_connector_id": "mca_EnqpwnxfB7GppsknYX7r",
    "incremental_authorization_allowed": false,
    "authorization_count": null,
    "incremental_authorizations": null,
    "external_authentication_details": null,
    "external_3ds_authentication_attempted": false,
    "expires_on": "2025-09-08T12:35:27.430Z",
    "fingerprint": null,
    "browser_info": {
        "os_type": "macOS",
        "language": "en-GB",
        "time_zone": -330,
        "os_version": "10.15.7",
        "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
        "color_depth": 24,
        "device_model": "Macintosh",
        "java_enabled": true,
        "screen_width": 1728,
        "accept_header": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "screen_height": 1117,
        "java_script_enabled": true
    },
    "payment_channel": null,
    "payment_method_id": "pm_49DfXLBrPJ1t90oFfmPi",
    "network_transaction_id": "PUKVC400BTCOGMT807X6ANT",
    "payment_method_status": "active",
    "updated": "2025-09-08T12:20:28.309Z",
    "split_payments": null,
    "frm_metadata": null,
    "extended_authorization_applied": null,
    "capture_before": null,
    "merchant_order_reference_id": null,
    "order_tax_amount": null,
    "connector_mandate_id": "524439809-536594893",
    "card_discovery": null,
    "force_3ds_challenge": false,
    "force_3ds_challenge_trigger": false,
    "issuer_error_code": null,
    "issuer_error_message": null,
    "is_iframe_redirection_enabled": null,
    "whole_connector_response": null,
    "enable_partial_authorization": null
}
```

MIT

```
curl --location 'http://localhost:8080/payments' \
--header 'Content-Type: application/json' \
--header 'Accept: application/json' \
--header 'api-key: dev_KE9tLklSMgmO0Wh3Zy8zu5IoKEEKynF16mY53eV84Dx5TbRjpHfSJuUt7WVEgd4r' \
--data '{
    "amount": 10000,
    "currency": "USD",
    "off_session": true,
    "confirm": true,
    "capture_method": "automatic",
    "recurring_details": {
        "type": "payment_method_id",
        "data": "pm_49DfXLBrPJ1t90oFfmPi"
    },
    
    "customer_id": "cus_12345",
    "connector": [
        "authorizedotnet"
    ]   
    
}'
```

Response
```
{
    "payment_id": "pay_vJpAQM8GUsOHKLCikEVP",
    "merchant_id": "postman_merchant_GHAction_5c96aafd-4990-44b3-b44d-c66b2a76f962",
    "status": "succeeded",
    "amount": 10000,
    "net_amount": 10000,
    "shipping_cost": null,
    "amount_capturable": 0,
    "amount_received": 10000,
    "connector": "authorizedotnet",
    "client_secret": "pay_vJpAQM8GUsOHKLCikEVP_secret_9P0sKSn1eekdRyBQJHu3",
    "created": "2025-09-08T12:22:01.910Z",
    "currency": "USD",
    "customer_id": "cus_12345",
    "customer": {
        "id": "cus_12345",
        "name": null,
        "email": null,
        "phone": null,
        "phone_country_code": null
    },
    "description": null,
    "refunds": null,
    "disputes": null,
    "mandate_id": null,
    "mandate_data": null,
    "setup_future_usage": null,
    "off_session": true,
    "capture_on": null,
    "capture_method": "automatic",
    "payment_method": "wallet",
    "payment_method_data": {
        "wallet": {
            "google_pay": {
                "last4": "1111",
                "card_network": "VISA",
                "type": "CARD"
            }
        },
        "billing": null
    },
    "payment_token": null,
    "shipping": null,
    "billing": null,
    "order_details": null,
    "email": null,
    "name": null,
    "phone": null,
    "return_url": null,
    "authentication_type": "no_three_ds",
    "statement_descriptor_name": null,
    "statement_descriptor_suffix": null,
    "next_action": null,
    "cancellation_reason": null,
    "error_code": null,
    "error_message": null,
    "unified_code": null,
    "unified_message": null,
    "payment_experience": null,
    "payment_method_type": "google_pay",
    "connector_label": null,
    "business_country": null,
    "business_label": "default",
    "business_sub_label": null,
    "allowed_payment_method_types": null,
    "ephemeral_key": {
        "customer_id": "cus_12345",
        "created_at": 1757334121,
        "expires": 1757337721,
        "secret": "epk_2cd756da4d794f4dac5a22e10457aa4c"
    },
    "manual_retry_allowed": false,
    "connector_transaction_id": "80044364565",
    "frm_message": null,
    "metadata": null,
    "connector_metadata": null,
    "feature_metadata": {
        "redirect_response": null,
        "search_tags": null,
        "apple_pay_recurring_details": null,
        "gateway_system": "direct"
    },
    "reference_id": "80044364565",
    "payment_link": null,
    "profile_id": "pro_ayYGLzXmcF3E9JgC7HZO",
    "surcharge_details": null,
    "attempt_count": 1,
    "merchant_decision": null,
    "merchant_connector_id": "mca_EnqpwnxfB7GppsknYX7r",
    "incremental_authorization_allowed": false,
    "authorization_count": null,
    "incremental_authorizations": null,
    "external_authentication_details": null,
    "external_3ds_authentication_attempted": false,
    "expires_on": "2025-09-08T12:37:01.910Z",
    "fingerprint": null,
    "browser_info": null,
    "payment_channel": null,
    "payment_method_id": "pm_49DfXLBrPJ1t90oFfmPi",
    "network_transaction_id": "D9LHUD6TDE3XP3JSXZTUEB2",
    "payment_method_status": "active",
    "updated": "2025-09-08T12:22:03.468Z",
    "split_payments": null,
    "frm_metadata": null,
    "extended_authorization_applied": null,
    "capture_before": null,
    "merchant_order_reference_id": null,
    "order_tax_amount": null,
    "connector_mandate_id": "524439809-536594893",
    "card_discovery": null,
    "force_3ds_challenge": false,
    "force_3ds_challenge_trigger": false,
    "issuer_error_code": null,
    "issuer_error_message": null,
    "is_iframe_redirection_enabled": null,
    "whole_connector_response": null,
    "enable_partial_authorization": null
}
```

</details>


## Checklist
<!-- Put an `x` in the boxes that apply -->

- [x] I formatted the code `cargo +nightly fmt --all`
- [x] I addressed lints thrown by `cargo clippy`
- [x] I reviewed the submitted code
- [ ] I added unit tests for my changes where possible

## Issue # Context


## Requirements
Based on the PR description and linked issue above, please implement the required changes.

**Please implement this feature.**
